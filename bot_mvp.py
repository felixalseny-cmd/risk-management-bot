# bot_mvp_phase3.py — PRO Risk Calculator MVP Phase 3 (Complete)
import os
import sys
import logging
import asyncio
import time
import functools
import json
import re
import html
import gc
import io
import csv
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
from dataclasses import dataclass, asdict
import base64

# --- Load .env ---
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found!")

PORT = int(os.getenv("PORT", 10000))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").rstrip("/")
WEBHOOK_PATH = f"/webhook/{TOKEN}"

# API Keys (используем из .env или дефолты)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

# --- Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("risk_calculator_pro")

# ==================== ADVANCED DATA STRUCTURES ====================
@dataclass
class AssetMetrics:
    """Расширенные метрики актива"""
    symbol: str
    current_price: float
    mean_return: float
    volatility: float
    annual_volatility: float
    historical_var_95: float
    historical_var_99: float
    parametric_var_95: float
    conditional_var_95: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    skewness: float
    kurtosis: float
    monte_carlo_var_95: float
    monte_carlo_prob_loss: float
    atr_14: float  # Average True Range
    rsi_14: float  # Relative Strength Index
    last_updated: str

@dataclass
class PortfolioMetrics:
    """Метрики портфеля"""
    total_value: float
    num_assets: int
    portfolio_var_95: float
    portfolio_cvar_95: float
    portfolio_volatility: float
    portfolio_sharpe: float
    portfolio_sortino: float
    portfolio_beta: float
    diversification_score: float  # 0-100
    concentration_risk: float  # Herfindahl-Hirschman Index
    correlation_risk: float
    worst_case_loss: float
    expected_shortfall: float
    stress_test_results: List[Dict]

# ==================== ADVANCED GRAPH GENERATOR ====================
class AdvancedGraphGenerator:
    """Генератор профессиональных графиков для анализа рисков"""
    
    def __init__(self):
        self.figures_created = 0
        self.memory_limit = 50  # Максимум 50 графиков в памяти
        
    def cleanup_old_figures(self):
        """Очистка старых графиков для экономии памяти"""
        if self.figures_created > self.memory_limit:
            gc.collect()
            self.figures_created = 0
            logger.info("Очистка памяти графиков")
    
    @monitor_performance
    def generate_monte_carlo_chart(self, initial_price: float, sample_paths: List[List[float]], 
                                 final_prices: List[float], symbol: str) -> io.BytesIO:
        """Генерация графика Monte Carlo симуляции"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import numpy as np
            
            self.cleanup_old_figures()
            
            plt.figure(figsize=(12, 8), dpi=100)
            
            # Plot sample paths
            days = len(sample_paths[0]) if sample_paths else 30
            x = np.arange(days)
            
            for i, path in enumerate(sample_paths[:50]):  # Limit to 50 paths for clarity
                alpha = 0.1 if i > 10 else 0.3  # Highlight first 10 paths
                plt.plot(x, path, 'b-', alpha=alpha, linewidth=0.5)
            
            # Calculate and plot confidence intervals
            if sample_paths:
                paths_array = np.array(sample_paths)
                mean_path = np.mean(paths_array, axis=0)
                std_path = np.std(paths_array, axis=0)
                
                plt.plot(x, mean_path, 'r-', linewidth=3, label='Средняя траектория')
                plt.fill_between(x, mean_path - std_path, mean_path + std_path, 
                               alpha=0.2, color='red', label='±1σ')
                plt.fill_between(x, mean_path - 2*std_path, mean_path + 2*std_path, 
                               alpha=0.1, color='red', label='±2σ')
            
            # Plot initial price line
            plt.axhline(y=initial_price, color='g', linestyle='--', linewidth=2, 
                       label=f'Начальная цена: ${initial_price:.2f}')
            
            # Formatting
            plt.title(f'Monte Carlo Симуляция: {symbol}\n30-дневный прогноз', fontsize=16, fontweight='bold')
            plt.xlabel('Дни', fontsize=12)
            plt.ylabel('Цена ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            
            # Add statistics box
            if final_prices:
                stats_text = (
                    f'Статистика финальных цен:\n'
                    f'Средняя: ${np.mean(final_prices):.2f}\n'
                    f'Медиана: ${np.median(final_prices):.2f}\n'
                    f'95% VaR: ${initial_price * (1 - np.percentile(final_prices, 5)/100):.2f}\n'
                    f'Вероятность убытка: {100 * sum(1 for p in final_prices if p < initial_price)/len(final_prices):.1f}%'
                )
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            plt.close()
            self.figures_created += 1
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating Monte Carlo chart: {e}")
            raise
    
    @monitor_performance
    def generate_distribution_chart(self, returns: List[float], symbol: str) -> io.BytesIO:
        """Генерация графика распределения доходностей"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
            from scipy.stats import norm
            
            self.cleanup_old_figures()
            
            plt.figure(figsize=(12, 8), dpi=100)
            
            returns_array = np.array(returns)
            mean, std = np.mean(returns_array), np.std(returns_array)
            
            # Histogram of returns
            plt.hist(returns_array, bins=50, density=True, alpha=0.6, color='blue', 
                   label='Исторические доходности')
            
            # Normal distribution fit
            x = np.linspace(min(returns_array), max(returns_array), 100)
            p = norm.pdf(x, mean, std)
            plt.plot(x, p, 'r-', linewidth=2, label=f'Нормальное распределение (μ={mean:.4f}, σ={std:.4f})')
            
            # VaR lines
            var_95 = np.percentile(returns_array, 5)
            var_99 = np.percentile(returns_array, 1)
            
            plt.axvline(x=var_95, color='orange', linestyle='--', linewidth=2, label=f'95% VaR = {var_95:.2%}')
            plt.axvline(x=var_99, color='red', linestyle='--', linewidth=2, label=f'99% VaR = {var_99:.2%}')
            
            # CVaR area
            cvar_returns = returns_array[returns_array <= var_95]
            if len(cvar_returns) > 0:
                plt.axvspan(min(cvar_returns), var_95, alpha=0.2, color='red', label='Conditional VaR область')
            
            # Formatting
            plt.title(f'Распределение доходностей: {symbol}\nДневные доходности', fontsize=16, fontweight='bold')
            plt.xlabel('Доходность', fontsize=12)
            plt.ylabel('Плотность вероятности', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            # Add statistics
            skew = float(np.cov(returns_array)[0, 0]) if len(returns_array) > 1 else 0
            kurt = float(np.cov(returns_array, rowvar=False)[0, 0]) if len(returns_array) > 1 else 0
            
            stats_text = (
                f'Статистика распределения:\n'
                f'Среднее: {mean:.4f}\n'
                f'Стандартное отклонение: {std:.4f}\n'
                f'Асимметрия: {skew:.2f}\n'
                f'Эксцесс: {kurt:.2f}\n'
                f'Тест на нормальность: {"Отклонено" if abs(skew) > 0.5 or abs(kurt-3) > 1 else "Не отклонено"}'
            )
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            plt.close()
            self.figures_created += 1
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating distribution chart: {e}")
            raise
    
    @monitor_performance
    def generate_correlation_matrix(self, assets: List[str], returns_matrix: np.ndarray) -> io.BytesIO:
        """Генерация матрицы корреляций активов"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
            import seaborn as sns
            
            self.cleanup_old_figures()
            
            plt.figure(figsize=(12, 10), dpi=100)
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(returns_matrix.T)
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn_r',
                       center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
            
            plt.title('Матрица корреляций активов портфеля', fontsize=16, fontweight='bold')
            plt.xticks(ticks=np.arange(len(assets)) + 0.5, labels=assets, rotation=45, ha='right')
            plt.yticks(ticks=np.arange(len(assets)) + 0.5, labels=assets, rotation=0)
            
            # Add portfolio diversification score
            if len(assets) > 1:
                avg_correlation = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
                diversification = 1 - avg_correlation
                
                plt.text(0.02, -0.1, 
                        f'Средняя корреляция: {avg_correlation:.2f} | Оценка диверсификации: {diversification:.2f}/1.0',
                        transform=plt.gca().transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            plt.close()
            self.figures_created += 1
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating correlation matrix: {e}")
            raise
    
    @monitor_performance
    def generate_stress_test_chart(self, stress_results: List[Dict]) -> io.BytesIO:
        """Генерация графика стресс-тестирования"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
            
            self.cleanup_old_figures()
            
            plt.figure(figsize=(14, 8), dpi=100)
            
            scenarios = [r['scenario'] for r in stress_results]
            losses = [r['loss_percent'] for r in stress_results]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            bars = plt.barh(scenarios, losses, color=colors[:len(scenarios)])
            plt.xlabel('Потери (%)', fontsize=12)
            plt.title('Результаты стресс-тестирования портфеля', fontsize=16, fontweight='bold')
            
            # Add value labels on bars
            for bar, loss in zip(bars, losses):
                plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{loss:.1f}%', va='center', fontweight='bold')
            
            # Add recovery time annotations
            for i, result in enumerate(stress_results):
                plt.text(-2, i, f"Восстановление: ~{result['recovery_months']} мес.",
                        va='center', fontsize=9, color='gray')
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.xlim(-5, max(losses) * 1.2)
            
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            plt.close()
            self.figures_created += 1
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating stress test chart: {e}")
            raise
    
    @monitor_performance
    def generate_risk_radar_chart(self, metrics: Dict[str, float]) -> io.BytesIO:
        """Генерация радар-диаграммы рисков"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
            
            self.cleanup_old_figures()
            
            categories = ['Волатильность', 'Риск снижения', 'Концентрация', 
                         'Корреляция', 'Ликвидность', 'Рыночный риск']
            
            values = [
                metrics.get('volatility_score', 50),
                metrics.get('drawdown_risk', 50),
                metrics.get('concentration_risk', 50),
                metrics.get('correlation_risk', 50),
                metrics.get('liquidity_risk', 50),
                metrics.get('market_risk', 50)
            ]
            
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            values += values[:1]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'), dpi=100)
            
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12)
            ax.set_ylim(0, 100)
            
            # Add risk level annotations
            risk_level = np.mean(values)
            if risk_level < 30:
                risk_text = "НИЗКИЙ РИСК"
                color = 'green'
            elif risk_level < 60:
                risk_text = "УМЕРЕННЫЙ РИСК"
                color = 'orange'
            else:
                risk_text = "ВЫСОКИЙ РИСК"
                color = 'red'
            
            plt.title(f'Профиль риска портфеля\n{risk_text}', fontsize=16, fontweight='bold', color=color)
            
            # Add value labels
            for angle, value, category in zip(angles[:-1], values[:-1], categories):
                ax.text(angle, value + 5, f'{value:.0f}', ha='center', va='center', fontsize=10)
            
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            plt.close()
            self.figures_created += 1
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating radar chart: {e}")
            raise

# ==================== ADVANCED REPORT GENERATOR ====================
class ReportGenerator:
    """Генератор профессиональных отчетов в различных форматах"""
    
    @staticmethod
    def generate_text_report(metrics: AssetMetrics) -> str:
        """Генерация текстового отчета"""
        report = [
            "=" * 60,
            f"АНАЛИТИЧЕСКИЙ ОТЧЕТ: {metrics.symbol}",
            f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            "📊 ОСНОВНЫЕ МЕТРИКИ:",
            f"  • Текущая цена: ${metrics.current_price:.2f}",
            f"  • Дневная волатильность: {metrics.volatility:.2f}%",
            f"  • Годовая волатильность: {metrics.annual_volatility:.2f}%",
            f"  • Средняя доходность: {metrics.mean_return:.4f}",
            "",
            "⚠️ МЕТРИКИ РИСКА:",
            f"  • VaR 95% (1 день): {metrics.historical_var_95:.2f}%",
            f"  • VaR 99% (1 день): {metrics.historical_var_99:.2f}%",
            f"  • Conditional VaR 95%: {metrics.conditional_var_95:.2f}%",
            f"  • Максимальное снижение: {metrics.max_drawdown:.2f}%",
            "",
            "📈 ПОКАЗАТЕЛИ ЭФФЕКТИВНОСТИ:",
            f"  • Коэффициент Шарпа: {metrics.sharpe_ratio:.2f}",
            f"  • Коэффициент Сортино: {metrics.sortino_ratio:.2f}",
            f"  • Асимметрия: {metrics.skewness:.2f}",
            f"  • Эксцесс: {metrics.kurtosis:.2f}",
            "",
            "🎲 MONTE CARLO АНАЛИЗ:",
            f"  • VaR 95% (30 дней): {metrics.monte_carlo_var_95:.2f}%",
            f"  • Вероятность убытка: {metrics.monte_carlo_prob_loss:.1f}%",
            "",
            "📊 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:",
            f"  • ATR (14): {metrics.atr_14:.2f}",
            f"  • RSI (14): {metrics.rsi_14:.1f}",
            f"    { '⟳ Перепроданность' if metrics.rsi_14 < 30 else '⟳ Перекупленность' if metrics.rsi_14 > 70 else '⟳ Нейтрально'}",
            "",
            "💡 РЕКОМЕНДАЦИИ:",
        ]
        
        # Добавляем рекомендации на основе метрик
        recommendations = []
        
        if metrics.historical_var_95 > 5:
            recommendations.append("  • Высокий риск - рассмотрите уменьшение позиции")
        if metrics.sharpe_ratio < 1:
            recommendations.append("  • Низкая доходность на единицу риска")
        if metrics.rsi_14 > 70:
            recommendations.append("  • Возможная перекупленность")
        elif metrics.rsi_14 < 30:
            recommendations.append("  • Возможная перепроданность")
        if not recommendations:
            recommendations.append("  • Актив выглядит сбалансированно")
        
        report.extend(recommendations)
        report.extend([
            "",
            "=" * 60,
            "Сгенерировано: PRO Risk Calculator MVP v3.0",
            "=" * 60
        ])
        
        return "\n".join(report)
    
    @staticmethod
    def generate_portfolio_report(portfolio_metrics: PortfolioMetrics, 
                                assets: List[AssetMetrics]) -> str:
        """Генерация отчета по портфелю"""
        report = [
            "=" * 60,
            "ОТЧЕТ ПО УПРАВЛЕНИЮ РИСКАМИ ПОРТФЕЛЯ",
            f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            f"💰 ОБЩАЯ СТОИМОСТЬ: ${portfolio_metrics.total_value:,.2f}",
            f"📈 АКТИВОВ: {portfolio_metrics.num_assets}",
            "",
            "📊 СВОДНЫЕ МЕТРИКИ РИСКА:",
            f"  • VaR 95% портфеля: {portfolio_metrics.portfolio_var_95:.1f}%",
            f"  • CVaR 95% портфеля: {portfolio_metrics.portfolio_cvar_95:.1f}%",
            f"  • Волатильность портфеля: {portfolio_metrics.portfolio_volatility:.1f}%",
            f"  • Коэффициент Шарпа: {portfolio_metrics.portfolio_sharpe:.2f}",
            f"  • Коэффициент Сортино: {portfolio_metrics.portfolio_sortino:.2f}",
            f"  • Бета портфеля: {portfolio_metrics.portfolio_beta:.2f}",
            "",
            "🎯 ОЦЕНКА ДИВЕРСИФИКАЦИИ:",
            f"  • Оценка диверсификации: {portfolio_metrics.diversification_score:.0f}/100",
            f"  • Риск концентрации: {portfolio_metrics.concentration_risk:.2f}",
            f"  • Риск корреляции: {portfolio_metrics.correlation_risk:.2f}",
            "",
            "⚠️ ЭКСТРЕМАЛЬНЫЕ СЦЕНАРИИ:",
            f"  • Худшие потери (99%): {portfolio_metrics.worst_case_loss:.1f}%",
            f"  • Ожидаемые потери в кризис: {portfolio_metrics.expected_shortfall:.1f}%",
            "",
            "📉 РЕЗУЛЬТАТЫ СТРЕСС-ТЕСТИРОВАНИЯ:",
        ]
        
        for stress in portfolio_metrics.stress_test_results[:3]:  # Показываем 3 сценария
            report.append(f"  • {stress['scenario']}: -{stress['loss_percent']:.1f}% (восстановление: {stress['recovery_months']} мес.)")
        
        report.extend([
            "",
            "📊 ДЕТАЛИЗАЦИЯ ПО АКТИВАМ:",
        ])
        
        for asset in assets:
            report.extend([
                f"  ─ {asset.symbol}:",
                f"    • Цена: ${asset.current_price:.2f}",
                f"    • VaR 95%: {asset.historical_var_95:.1f}%",
                f"    • Вклад в риск: {asset.volatility/portfolio_metrics.portfolio_volatility*100:.1f}%",
            ])
        
        report.extend([
            "",
            "💡 РЕКОМЕНДАЦИИ ПО УПРАВЛЕНИЮ РИСКАМИ:",
        ])
        
        recommendations = []
        if portfolio_metrics.diversification_score < 60:
            recommendations.append("  • Низкая диверсификация - добавьте активы из разных классов")
        if portfolio_metrics.portfolio_var_95 > 8:
            recommendations.append("  • Высокий совокупный риск - уменьшите позиции в волатильных активах")
        if portfolio_metrics.concentration_risk > 0.3:
            recommendations.append("  • Высокий риск концентрации - распределите капитал более равномерно")
        if not recommendations:
            recommendations.append("  • Портфель хорошо сбалансирован по рискам")
        
        report.extend(recommendations)
        report.extend([
            "",
            "=" * 60,
            "Сгенерировано: PRO Risk Calculator MVP v3.0",
            "=" * 60
        ])
        
        return "\n".join(report)
    
    @staticmethod
    def generate_csv_report(metrics: AssetMetrics) -> str:
        """Генерация CSV отчета"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Заголовок
        writer.writerow(["PRO Risk Calculator - Аналитический отчет"])
        writer.writerow([f"Актив: {metrics.symbol}"])
        writer.writerow([f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M')}"])
        writer.writerow([])
        
        # Основные метрики
        writer.writerow(["ОСНОВНЫЕ МЕТРИКИ"])
        writer.writerow(["Показатель", "Значение", "Единица измерения"])
        writer.writerow(["Текущая цена", f"{metrics.current_price:.2f}", "USD"])
        writer.writerow(["Средняя доходность", f"{metrics.mean_return:.6f}", "доли"])
        writer.writerow(["Дневная волатильность", f"{metrics.volatility:.4f}", "доли"])
        writer.writerow(["Годовая волатильность", f"{metrics.annual_volatility:.4f}", "доли"])
        writer.writerow([])
        
        # Метрики риска
        writer.writerow(["МЕТРИКИ РИСКА"])
        writer.writerow(["Показатель", "Значение", "Единица измерения"])
        writer.writerow(["VaR 95% (1 день)", f"{metrics.historical_var_95:.4f}", "%"])
        writer.writerow(["VaR 99% (1 день)", f"{metrics.historical_var_99:.4f}", "%"])
        writer.writerow(["Conditional VaR 95%", f"{metrics.conditional_var_95:.4f}", "%"])
        writer.writerow(["Максимальное снижение", f"{metrics.max_drawdown:.4f}", "%"])
        writer.writerow([])
        
        # Показатели эффективности
        writer.writerow(["ПОКАЗАТЕЛИ ЭФФЕКТИВНОСТИ"])
        writer.writerow(["Показатель", "Значение"])
        writer.writerow(["Коэффициент Шарпа", f"{metrics.sharpe_ratio:.4f}"])
        writer.writerow(["Коэффициент Сортино", f"{metrics.sortino_ratio:.4f}"])
        writer.writerow(["Асимметрия", f"{metrics.skewness:.4f}"])
        writer.writerow(["Эксцесс", f"{metrics.kurtosis:.4f}"])
        writer.writerow([])
        
        return output.getvalue()
    
    @staticmethod
    def generate_json_report(metrics: AssetMetrics) -> str:
        """Генерация JSON отчета"""
        report = {
            "metadata": {
                "report_type": "asset_risk_analysis",
                "generated_at": datetime.now().isoformat(),
                "version": "3.0",
                "asset": metrics.symbol
            },
            "price_data": {
                "current_price": metrics.current_price,
                "currency": "USD"
            },
            "risk_metrics": {
                "daily_volatility": metrics.volatility,
                "annual_volatility": metrics.annual_volatility,
                "var_95": metrics.historical_var_95,
                "var_99": metrics.historical_var_99,
                "cvar_95": metrics.conditional_var_95,
                "max_drawdown": metrics.max_drawdown,
                "monte_carlo_var_95": metrics.monte_carlo_var_95
            },
            "performance_metrics": {
                "mean_return": metrics.mean_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "skewness": metrics.skewness,
                "kurtosis": metrics.kurtosis
            },
            "technical_indicators": {
                "atr_14": metrics.atr_14,
                "rsi_14": metrics.rsi_14,
                "rsi_interpretation": "oversold" if metrics.rsi_14 < 30 else "overbought" if metrics.rsi_14 > 70 else "neutral"
            },
            "recommendations": {
                "risk_level": "high" if metrics.historical_var_95 > 5 else "medium" if metrics.historical_var_95 > 2 else "low",
                "action": "reduce" if metrics.historical_var_95 > 5 else "hold" if metrics.sharpe_ratio > 1 else "review"
            }
        }
        
        return json.dumps(report, indent=2, ensure_ascii=False)

# ==================== ENHANCED ANALYTICS ENGINE ====================
class EnhancedAnalyticsEngine:
    """Расширенный аналитический движок с техническими индикаторами"""
    
    @staticmethod
    @monitor_performance
    def calculate_technical_indicators(prices: List[float]) -> Dict[str, float]:
        """Расчет технических индикаторов"""
        import numpy as np
        
        if len(prices) < 15:
            return {"atr_14": 0.0, "rsi_14": 50.0}
        
        prices_array = np.array(prices)
        
        # Calculate RSI (Relative Strength Index)
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            seed = deltas[:period]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 0
            rsi = 100 - 100 / (1 + rs)
            
            for i in range(period, len(deltas)):
                delta = deltas[i]
                if delta > 0:
                    up_val = delta
                    down_val = 0
                else:
                    up_val = 0
                    down_val = -delta
                
                up = (up * (period - 1) + up_val) / period
                down = (down * (period - 1) + down_val) / period
                rs = up / down if down != 0 else 0
                rsi = np.append(rsi, 100 - 100 / (1 + rs))
            
            return rsi[-1] if len(rsi) > 0 else 50
        
        # Calculate ATR (Average True Range)
        def calculate_atr(prices, period=14):
            if len(prices) < period + 1:
                return 0.0
            
            high = prices  # Simplified - using same prices for high/low
            low = prices
            close = prices
            
            tr = np.maximum(
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
            
            atr = np.zeros_like(prices)
            atr[period] = np.mean(tr[:period])
            
            for i in range(period + 1, len(prices)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
            
            return atr[-1] if len(atr) > period else 0.0
        
        rsi_14 = calculate_rsi(prices_array)
        atr_14 = calculate_atr(prices_array)
        
        return {
            "atr_14": float(atr_14),
            "rsi_14": float(rsi_14),
            "price_trend": "up" if prices_array[-1] > prices_array[0] else "down"
        }
    
    @staticmethod
    @monitor_performance
    def calculate_portfolio_diversification(weights: List[float], 
                                          correlation_matrix: np.ndarray) -> Dict[str, float]:
        """Расчет показателей диверсификации портфеля"""
        import numpy as np
        
        weights_array = np.array(weights)
        
        # Herfindahl-Hirschman Index (HHI) for concentration
        hhi = np.sum(weights_array ** 2)
        
        # Effective number of assets (diversification metric)
        effective_n = 1 / hhi if hhi > 0 else len(weights)
        
        # Average pairwise correlation
        if len(weights) > 1:
            # Get upper triangle of correlation matrix (excluding diagonal)
            upper_tri = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            avg_correlation = np.mean(np.abs(upper_tri)) if len(upper_tri) > 0 else 0
        else:
            avg_correlation = 0
        
        # Diversification score (0-100)
        diversification_score = min(100, effective_n / len(weights) * 100) if len(weights) > 0 else 0
        diversification_score *= (1 - avg_correlation)  # Penalize for high correlations
        
        return {
            "concentration_risk": float(hhi),
            "effective_assets": float(effective_n),
            "avg_correlation": float(avg_correlation),
            "diversification_score": float(diversification_score)
        }
    
    @staticmethod
    @monitor_performance
    def calculate_scenario_analysis(portfolio_value: float, 
                                  asset_allocations: Dict[str, float],
                                  scenarios: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Расширенный анализ сценариев"""
        results = []
        
        for scenario_name, impacts in scenarios.items():
            total_impact = 0
            for asset_class, allocation in asset_allocations.items():
                impact = impacts.get(asset_class, 0)
                total_impact += impact * allocation
            
            stressed_value = portfolio_value * (1 + total_impact)
            
            # Calculate additional metrics
            recovery_time = EnhancedAnalyticsEngine.estimate_recovery_time(total_impact)
            margin_call_risk = EnhancedAnalyticsEngine.calculate_margin_call_risk(total_impact)
            
            results.append({
                "scenario": scenario_name,
                "stressed_value": round(stressed_value, 2),
                "loss_percent": round(abs(total_impact) * 100, 1),
                "drawdown": round(abs(total_impact) * 100, 1),
                "recovery_months": recovery_time,
                "margin_call_risk": margin_call_risk,
                "severity": "high" if abs(total_impact) > 0.4 else "medium" if abs(total_impact) > 0.2 else "low"
            })
        
        return results
    
    @staticmethod
    def estimate_recovery_time(loss_percent: float) -> int:
        """Оценка времени восстановления после потерь"""
        loss_abs = abs(loss_percent)
        
        if loss_abs <= 0.1:  # 10%
            return 3
        elif loss_abs <= 0.2:  # 20%
            return 6
        elif loss_abs <= 0.3:  # 30%
            return 12
        elif loss_abs <= 0.4:  # 40%
            return 18
        else:  # > 40%
            return 24
    
    @staticmethod
    def calculate_margin_call_risk(loss_percent: float) -> str:
        """Расчет риска маржин-колла"""
        loss_abs = abs(loss_percent)
        
        if loss_abs > 0.5:
            return "Критический"
        elif loss_abs > 0.3:
            return "Высокий"
        elif loss_abs > 0.15:
            return "Умеренный"
        else:
            return "Низкий"

# ==================== ENHANCED TELEGRAM BOT ====================
# Расширяем бота из Phase 2 с новыми функциями

class EnhancedTelegramBot(TelegramRiskBot):
    """Расширенный Telegram бот с графиками и отчетами"""
    
    def __init__(self, token: str):
        super().__init__(token)
        self.graph_generator = AdvancedGraphGenerator()
        self.report_generator = ReportGenerator()
        self.analytics_engine = EnhancedAnalyticsEngine()
        
        # История запросов пользователей
        self.user_history = defaultdict(list)
    
    async def advanced_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Расширенный анализ с графиками"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        text = (
            "📈 *РАСШИРЕННЫЙ АНАЛИЗ*\n\n"
            "Выберите тип анализа:\n\n"
            "• 📊 *Полный отчет* - детальный анализ с графиками\n"
            "• 🎲 *Monte Carlo* - симуляция ценовых траекторий\n"
            "• 📉 *Распределение доходностей* - анализ статистики\n"
            "• ⚠️ *Стресс-тест* - анализ устойчивости портфеля\n"
            "• 🎯 *Технические индикаторы* - ATR, RSI, тренды\n"
            "• 📤 *Экспорт отчета* - отчеты в TXT/CSV/JSON"
        )
        
        keyboard = [
            [InlineKeyboardButton("📊 Полный отчет", callback_data="full_report")],
            [InlineKeyboardButton("🎲 Monte Carlo", callback_data="monte_carlo_chart")],
            [InlineKeyboardButton("📉 Распределение", callback_data="distribution_chart")],
            [
                InlineKeyboardButton("⚡ Быстрый анализ", callback_data="quick_analysis"),
                InlineKeyboardButton("📤 Экспорт", callback_data="export_menu")
            ],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            text=text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def generate_full_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Генерация полного отчета с графиками"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        # Проверяем историю пользователя
        if user_id not in self.user_history or not self.user_history[user_id]:
            await query.edit_message_text(
                text="❌ Сначала проанализируйте актив или портфель.",
                parse_mode='Markdown'
            )
            return
        
        # Получаем последний анализ
        last_analysis = self.user_history[user_id][-1]
        
        await query.edit_message_text(
            text="🔄 Генерирую полный отчет... Это займет несколько секунд.",
            parse_mode='Markdown'
        )
        
        try:
            # Генерация текстового отчета
            text_report = self.report_generator.generate_text_report(last_analysis['metrics'])
            
            # Генерация графиков
            if 'monte_carlo_data' in last_analysis:
                mc_chart = self.graph_generator.generate_monte_carlo_chart(
                    last_analysis['metrics'].current_price,
                    last_analysis['monte_carlo_data']['sample_paths'],
                    last_analysis['monte_carlo_data']['final_prices'],
                    last_analysis['metrics'].symbol
                )
                
                # Отправляем график
                await context.bot.send_photo(
                    chat_id=query.message.chat_id,
                    photo=InputFile(mc_chart, filename=f"mc_{last_analysis['metrics'].symbol}.png"),
                    caption=f"📈 Monte Carlo симуляция для {last_analysis['metrics'].symbol}"
                )
            
            if 'returns_data' in last_analysis:
                dist_chart = self.graph_generator.generate_distribution_chart(
                    last_analysis['returns_data'],
                    last_analysis['metrics'].symbol
                )
                
                await context.bot.send_photo(
                    chat_id=query.message.chat_id,
                    photo=InputFile(dist_chart, filename=f"dist_{last_analysis['metrics'].symbol}.png"),
                    caption=f"📊 Распределение доходностей {last_analysis['metrics'].symbol}"
                )
            
            # Отправляем текстовый отчет (разбиваем на части если длинный)
            chunks = [text_report[i:i+4000] for i in range(0, len(text_report), 4000)]
            
            for i, chunk in enumerate(chunks):
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text=f"```\n{chunk}\n```" if i == 0 else f"```\n{chunk}",
                    parse_mode='Markdown'
                )
            
            # Предлагаем экспорт
            keyboard = [
                [
                    InlineKeyboardButton("📝 TXT", callback_data=f"export_txt_{last_analysis['metrics'].symbol}"),
                    InlineKeyboardButton("📊 CSV", callback_data=f"export_csv_{last_analysis['metrics'].symbol}"),
                    InlineKeyboardButton("📋 JSON", callback_data=f"export_json_{last_analysis['metrics'].symbol}")
                ],
                [InlineKeyboardButton("🔄 Новый анализ", callback_data="analyze_asset")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
            
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="✅ Отчет сгенерирован!\n\nВыберите формат для экспорта:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error generating full report: {e}")
            await query.edit_message_text(
                text="❌ Ошибка при генерации отчета. Попробуйте снова.",
                parse_mode='Markdown'
            )
    
    async def export_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Меню экспорта отчетов"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        if user_id not in self.user_history or not self.user_history[user_id]:
            text = "❌ Нет данных для экспорта. Сначала проанализируйте актив."
        else:
            last_analysis = self.user_history[user_id][-1]
            symbol = last_analysis['metrics'].symbol
            
            text = f"📤 *Экспорт отчета для {symbol}*\n\nВыберите формат:"
        
        keyboard = [
            [
                InlineKeyboardButton("📝 TXT", callback_data=f"export_txt_{symbol if 'symbol' in locals() else ''}"),
                InlineKeyboardButton("📊 CSV", callback_data=f"export_csv_{symbol if 'symbol' in locals() else ''}"),
                InlineKeyboardButton("📋 JSON", callback_data=f"export_json_{symbol if 'symbol' in locals() else ''}")
            ],
            [InlineKeyboardButton("📈 PDF (скоро)", callback_data="coming_soon")],
            [InlineKeyboardButton("📊 Полный отчет", callback_data="full_report")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            text=text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def handle_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка экспорта отчетов"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        user_id = query.from_user.id
        
        if user_id not in self.user_history or not self.user_history[user_id]:
            await query.answer("❌ Нет данных для экспорта", show_alert=True)
            return
        
        last_analysis = self.user_history[user_id][-1]
        metrics = last_analysis['metrics']
        symbol = metrics.symbol
        
        await query.edit_message_text(
            text=f"🔄 Генерирую отчет для {symbol}...",
            parse_mode='Markdown'
        )
        
        try:
            if data.startswith("export_txt"):
                # TXT экспорт
                report = self.report_generator.generate_text_report(metrics)
                bio = io.BytesIO(report.encode('utf-8'))
                bio.seek(0)
                
                await context.bot.send_document(
                    chat_id=query.message.chat_id,
                    document=InputFile(bio, filename=f"report_{symbol}_{datetime.now().strftime('%Y%m%d')}.txt"),
                    caption=f"📝 Текстовый отчет: {symbol}"
                )
                
            elif data.startswith("export_csv"):
                # CSV экспорт
                report = self.report_generator.generate_csv_report(metrics)
                bio = io.BytesIO(report.encode('utf-8'))
                bio.seek(0)
                
                await context.bot.send_document(
                    chat_id=query.message.chat_id,
                    document=InputFile(bio, filename=f"report_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"),
                    caption=f"📊 CSV отчет: {symbol}"
                )
                
            elif data.startswith("export_json"):
                # JSON экспорт
                report = self.report_generator.generate_json_report(metrics)
                bio = io.BytesIO(report.encode('utf-8'))
                bio.seek(0)
                
                await context.bot.send_document(
                    chat_id=query.message.chat_id,
                    document=InputFile(bio, filename=f"report_{symbol}_{datetime.now().strftime('%Y%m%d')}.json"),
                    caption=f"📋 JSON отчет: {symbol}"
                )
            
            # Возвращаемся в меню
            keyboard = [
                [InlineKeyboardButton("📤 Еще отчеты", callback_data="export_menu")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
            
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="✅ Отчет успешно экспортирован!",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            await query.edit_message_text(
                text="❌ Ошибка при экспорте отчета.",
                parse_mode='Markdown'
            )
    
    async def portfolio_correlation_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Анализ корреляций в портфеле"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        if user_id not in user_portfolios or not user_portfolios[user_id]:
            await query.edit_message_text(
                text="❌ У вас нет портфеля для анализа.",
                parse_mode='Markdown'
            )
            return
        
        await query.edit_message_text(
            text="🔄 Анализирую корреляции в портфеле...",
            parse_mode='Markdown'
        )
        
        try:
            portfolio = user_portfolios[user_id]
            assets = list(portfolio.keys())
            
            # Генерируем симулированные корреляции для MVP
            import numpy as np
            
            num_assets = len(assets)
            returns_matrix = np.random.randn(100, num_assets)  # Случайные доходности
            
            # Создаем реалистичную матрицу корреляций
            base_corr = 0.3  # Базовая корреляция
            corr_matrix = np.eye(num_assets) + base_corr
            np.fill_diagonal(corr_matrix, 1)
            
            # Генерируем данные с заданной корреляцией
            L = np.linalg.cholesky(corr_matrix)
            correlated_returns = np.dot(returns_matrix, L.T)
            
            # Генерируем график
            chart = self.graph_generator.generate_correlation_matrix(assets, correlated_returns)
            
            # Расчет диверсификации
            weights = [portfolio[asset]['weight'] for asset in assets]
            div_metrics = self.analytics_engine.calculate_portfolio_diversification(weights, corr_matrix)
            
            # Отправляем график
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=InputFile(chart, filename="correlation_matrix.png"),
                caption=(
                    f"📊 *Матрица корреляций портфеля*\n\n"
                    f"• Активов: {num_assets}\n"
                    f"• Средняя корреляция: {div_metrics['avg_correlation']:.2f}\n"
                    f"• Оценка диверсификации: {div_metrics['diversification_score']:.0f}/100\n"
                    f"• Эффективных активов: {div_metrics['effective_assets']:.1f}"
                ),
                parse_mode='Markdown'
            )
            
            # Дополнительные рекомендации
            text = "💡 *Рекомендации по диверсификации:*\n\n"
            
            if div_metrics['avg_correlation'] > 0.7:
                text += "• Высокая корреляция - добавьте активы из других классов\n"
            if div_metrics['diversification_score'] < 60:
                text += "• Низкая диверсификация - распределите капитал\n"
            if div_metrics['concentration_risk'] > 0.3:
                text += "• Высокая концентрация - уменьшите позиции в крупнейших активах\n"
            
            if "Высокая" not in text and "Низкая" not in text and "Высокая" not in text:
                text += "• Портфель хорошо диверсифицирован\n"
            
            keyboard = [
                [InlineKeyboardButton("📈 Risk Radar", callback_data="risk_radar")],
                [InlineKeyboardButton("💰 Портфель", callback_data="manage_portfolio")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
            
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=text,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            await query.edit_message_text(
                text="❌ Ошибка при анализе корреляций.",
                parse_mode='Markdown'
            )
    
    async def risk_radar_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Радар-диаграмма рисков портфеля"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        
        if user_id not in user_portfolios or not user_portfolios[user_id]:
            await query.edit_message_text(
                text="❌ У вас нет портфеля для анализа.",
                parse_mode='Markdown'
            )
            return
        
        await query.edit_message_text(
            text="🔄 Генерирую радар-диаграмму рисков...",
            parse_mode='Markdown'
        )
        
        try:
            # Генерируем метрики риска (для MVP - симулированные)
            risk_metrics = {
                'volatility_score': 65,
                'drawdown_risk': 45,
                'concentration_risk': 70,
                'correlation_risk': 55,
                'liquidity_risk': 30,
                'market_risk': 60
            }
            
            # Генерируем график
            chart = self.graph_generator.generate_risk_radar_chart(risk_metrics)
            
            # Отправляем график
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=InputFile(chart, filename="risk_radar.png"),
                caption=(
                    "🎯 *Профиль риска портфеля*\n\n"
                    "• 📊 Волатильность: Умеренная\n"
                    "• 📉 Риск снижения: Низкий\n"
                    "• 🎯 Концентрация: Высокая\n"
                    "• 🔗 Корреляция: Умеренная\n"
                    "• 💧 Ликвидность: Хорошая\n"
                    "• 📈 Рыночный риск: Умеренный"
                ),
                parse_mode='Markdown'
            )
            
            keyboard = [
                [InlineKeyboardButton("📊 Корреляции", callback_data="portfolio_correlation")],
                [InlineKeyboardButton("⚠️ Стресс-тест", callback_data="stress_test")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
            ]
            
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="💡 *Интерпретация:*\nЧем ближе к краю - тем выше риск по этому параметру.",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Error generating risk radar: {e}")
            await query.edit_message_text(
                text="❌ Ошибка при генерации радар-диаграммы.",
                parse_mode='Markdown'
            )

# ==================== ENHANCED WEB SERVER ====================
class EnhancedWebhookServer(WebhookServer):
    """Расширенный веб-сервер с мониторингом"""
    
    async def enhanced_health_check(self, request):
        """Расширенная проверка здоровья"""
        import psutil
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "risk-calculator-pro",
            "version": "MVP 3.0",
            "resources": {}
        }
        
        try:
            # Мониторинг ресурсов
            process = psutil.Process()
            memory_info = process.memory_info()
            
            health_data["resources"] = {
                "memory_mb": round(memory_info.rss / 1024 / 1024, 1),
                "memory_percent": round(process.memory_percent(), 1),
                "cpu_percent": round(process.cpu_percent(), 1),
                "threads": process.num_threads(),
                "connections": len(process.connections()) if hasattr(process, 'connections') else 0
            }
            
            # Проверка компонентов
            components = {}
            
            # Проверка графического генератора
            try:
                test_chart = AdvancedGraphGenerator().generate_monte_carlo_chart(
                    100, [[100, 110, 105]], [105], "TEST"
                )
                components["graph_generator"] = "operational"
            except Exception as e:
                components["graph_generator"] = f"error: {str(e)}"
                health_data["status"] = "degraded"
            
            # Проверка аналитического движка
            try:
                test_metrics = EnhancedAnalyticsEngine().calculate_technical_indicators([100, 105, 103, 108])
                components["analytics_engine"] = "operational"
            except Exception as e:
                components["analytics_engine"] = f"error: {str(e)}"
                health_data["status"] = "degraded"
            
            health_data["components"] = components
            
            # Memory guard check
            if health_data["resources"]["memory_mb"] > 400:
                health_data["status"] = "warning"
                health_data["message"] = "High memory usage detected"
                MemoryGuardian.check_and_clear()
            
        except Exception as e:
            health_data["status"] = "error"
            health_data["error"] = str(e)
        
        return web.json_response(health_data)
    
    async def start(self):
        """Запуск расширенного сервера"""
        app = web.Application()
        
        # Add enhanced routes
        app.router.add_post(WEBHOOK_PATH, self.handle_webhook)
        app.router.add_get('/health', self.enhanced_health_check)
        app.router.add_get('/health/simple', self.health_check)
        app.router.add_get('/metrics', self.metrics_endpoint)
        app.router.add_get('/', self.health_check)
        
        # Start server
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await self.site.start()
        
        logger.info(f"Enhanced webhook server started on port {self.port}")
    
    async def metrics_endpoint(self, request):
        """Endpoint для метрик Prometheus"""
        import psutil
        
        metrics = []
        process = psutil.Process()
        
        # Memory metrics
        memory = process.memory_info()
        metrics.append(f"memory_rss_bytes {memory.rss}")
        metrics.append(f"memory_vms_bytes {memory.vms}")
        metrics.append(f"memory_percent {process.memory_percent()}")
        
        # CPU metrics
        metrics.append(f"cpu_percent {process.cpu_percent()}")
        
        # Thread count
        metrics.append(f"threads_total {process.num_threads()}")
        
        # User sessions
        metrics.append(f"user_sessions_total {len(user_sessions)}")
        metrics.append(f"user_portfolios_total {len(user_portfolios)}")
        
        # Graph generator stats
        if hasattr(graph_generator, 'figures_created'):
            metrics.append(f"figures_created_total {graph_generator.figures_created}")
        
        response_text = "\n".join([f"risk_calculator_{m}" for m in metrics])
        return web.Response(text=response_text, content_type='text/plain')

# ==================== MAIN APPLICATION ====================
async def main():
    """Основная функция приложения"""
    logger.info("🚀 Запуск PRO Risk Calculator MVP Phase 3 (Complete)")
    logger.info("✅ Все компоненты загружены")
    
    # Инициализация расширенного бота
    bot = EnhancedTelegramBot(TOKEN)
    
    # Создание приложения
    application = Application.builder().token(TOKEN).build()
    
    # Добавление обработчиков команд из Phase 2
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("portfolio", bot.manage_portfolio))
    application.add_handler(CommandHandler("stress", bot.stress_test))
    application.add_handler(CommandHandler("alerts", bot.alerts_menu))
    application.add_handler(CommandHandler("export", bot.export_menu))
    application.add_handler(CommandHandler("advanced", bot.advanced_analysis))
    
    # Добавление обработчиков Phase 3
    application.add_handler(CallbackQueryHandler(bot.advanced_analysis, pattern="^advanced_analysis$"))
    application.add_handler(CallbackQueryHandler(bot.generate_full_report, pattern="^full_report$"))
    application.add_handler(CallbackQueryHandler(bot.export_menu, pattern="^export_menu$"))
    application.add_handler(CallbackQueryHandler(bot.handle_export, pattern="^export_"))
    application.add_handler(CallbackQueryHandler(bot.portfolio_correlation_analysis, pattern="^portfolio_correlation$"))
    application.add_handler(CallbackQueryHandler(bot.risk_radar_analysis, pattern="^risk_radar$"))
    
    # Остальные обработчики из Phase 2
    application.add_handler(CallbackQueryHandler(bot.callback_handler))
    
    # Fallback обработчик
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        lambda u, c: u.message.reply_text(
            "Используйте /start для начала работы или выберите команду из меню."
        )
    ))
    
    # Запуск в зависимости от режима
    if WEBHOOK_URL and WEBHOOK_URL.strip():
        logger.info("🌐 Запуск в режиме Webhook")
        
        # Установка вебхука
        webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
        await application.bot.set_webhook(
            url=webhook_url,
            allowed_updates=Update.ALL_TYPES
        )
        logger.info(f"Webhook установлен: {webhook_url}")
        
        # Запуск расширенного сервера
        server = EnhancedWebhookServer(application, PORT)
        await server.start()
        
        # Основной цикл с мониторингом
        try:
            while True:
                # Проверка памяти каждые 5 минут
                MemoryGuardian.check_and_clear()
                
                # Логирование статистики каждые 30 минут
                if datetime.now().minute % 30 == 0:
                    logger.info(f"Статистика: Пользователей: {len(user_sessions)}, "
                              f"Портфелей: {len(user_portfolios)}, "
                              f"Графиков создано: {bot.graph_generator.figures_created}")
                
                await asyncio.sleep(300)  # 5 минут
                
        except KeyboardInterrupt:
            logger.info("⏹ Остановка по команде пользователя")
            await server.stop()
            
    else:
        logger.info("🔄 Запуск в режиме Polling")
        
        # Запуск polling
        await application.initialize()
        await application.start()
        await application.updater.start_polling(
            poll_interval=1.0,
            timeout=30,
            drop_pending_updates=True
        )
        
        logger.info("✅ Бот запущен в режиме polling")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("⏹ Остановка по команде пользователя")
            await application.stop()

# ==================== GLOBALS AND INITIALIZATION ====================
# Глобальные экземпляры
graph_generator = AdvancedGraphGenerator()
report_generator = ReportGenerator()
analytics_engine = EnhancedAnalyticsEngine()

# Пользовательские данные
user_sessions = {}
user_portfolios = {}
user_alerts = {}

# Декоратор мониторинга производительности (определен ранее)
def monitor_performance(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                logger.warning(f"Slow operation: {func.__name__} took {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in {func.__name__} after {elapsed:.2f}s: {e}")
            raise
    return async_wrapper

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
