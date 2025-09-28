"""
Personal Trading Dashboard
Real-time monitoring and control interface
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
import argparse
import os

# Import personal trading system
from personal_trader import PersonalTradingSystem, TradingSignal
from backtest_engine import Backtester, momentum_strategy, mean_reversion_strategy

class TradingDashboard:
    def __init__(self, system: PersonalTradingSystem):
        self.system = system
        self.console = Console()
        self.running = True
        
    def create_layout(self) -> Layout:
        """Create dashboard layout"""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="signals", ratio=2),
            Layout(name="sidebar")
        )
        
        layout["sidebar"].split(
            Layout(name="portfolio"),
            Layout(name="metrics"),
            Layout(name="risk")
        )
        
        return layout
    
    def render_header(self) -> Panel:
        """Render header with system status"""
        status = "🟢 ACTIVE" if self.running else "🔴 STOPPED"
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        header_text = Text()
        header_text.append("PERSONAL TRADING SYSTEM", style="bold white")
        header_text.append(f"  |  Status: {status}")
        header_text.append(f"  |  {time_str}")
        
        return Panel(header_text, style="bold blue")
    
    def render_signals(self, signals: List[TradingSignal]) -> Panel:
        """Render trading signals table"""
        table = Table(title="Active Signals", show_lines=True)
        
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Action", width=6)
        table.add_column("Price", justify="right", width=10)
        table.add_column("Confidence", justify="right", width=10)
        table.add_column("Strategy", width=15)
        table.add_column("R:R", justify="right", width=6)
        table.add_column("Size", justify="right", width=8)
        
        for signal in signals[:10]:  # Show top 10
            action_style = "green" if signal.action == "BUY" else "red"
            conf_style = "green" if signal.confidence > 0.7 else "yellow" if signal.confidence > 0.6 else "white"
            
            table.add_row(
                signal.symbol,
                Text(signal.action, style=action_style),
                f"${signal.entry_price:.2f}",
                Text(f"{signal.confidence:.1%}", style=conf_style),
                signal.strategy[:15],
                f"{signal.risk_reward:.1f}",
                f"{signal.position_size:.1%}"
            )
        
        return Panel(table)
    
    def render_portfolio(self) -> Panel:
        """Render portfolio positions"""
        table = Table(title="Open Positions", show_lines=False)
        
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Shares", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        
        for symbol, position in list(self.system.portfolio.positions.items())[:5]:
            pnl = position.get('unrealized_pnl', 0)
            pnl_style = "green" if pnl > 0 else "red" if pnl < 0 else "white"
            
            table.add_row(
                symbol,
                str(position['shares']),
                f"${position['entry_price']:.2f}",
                f"${position.get('current_price', position['entry_price']):.2f}",
                Text(f"${pnl:.2f}", style=pnl_style)
            )
        
        if not self.system.portfolio.positions:
            table.add_row("No positions", "-", "-", "-", "-")
        
        return Panel(table)
    
    def render_metrics(self) -> Panel:
        """Render performance metrics"""
        metrics = self.system.get_portfolio_metrics()
        
        content = Text()
        content.append("Portfolio Metrics\n\n", style="bold")
        
        total_value = metrics.get('total_value', 100000)
        total_return = metrics.get('total_return_pct', 0)
        
        content.append(f"Total Value: ${total_value:,.2f}\n")
        content.append(f"Cash: ${metrics.get('cash', 0):,.2f}\n")
        content.append(f"Positions: {metrics.get('positions', 0)}\n")
        
        return_style = "green" if total_return > 0 else "red" if total_return < 0 else "white"
        content.append(f"Total Return: ", style="white")
        content.append(f"{total_return:.2f}%\n", style=return_style)
        
        content.append(f"Win Rate: {metrics.get('win_rate', 0):.1%}\n")
        content.append(f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}\n")
        
        return Panel(content)
    
    def render_risk(self) -> Panel:
        """Render risk metrics"""
        risk_status = self.system.risk_manager.check_risk_limits()
        
        content = Text()
        content.append("Risk Management\n\n", style="bold")
        
        risk_score = risk_status.get('risk_score', 0)
        risk_style = "green" if risk_score < 0.3 else "yellow" if risk_score < 0.6 else "red"
        
        content.append(f"Risk Score: ", style="white")
        content.append(f"{risk_score:.1%}\n", style=risk_style)
        
        # Risk limits
        limits = self.system.portfolio.risk_limits
        content.append(f"\nLimits:\n", style="bold")
        content.append(f"Max Position: {limits['max_position_size']:.0%}\n")
        content.append(f"Max Drawdown: {limits['max_drawdown']:.0%}\n")
        content.append(f"Daily Loss: {limits['daily_loss_limit']:.0%}\n")
        
        warnings = risk_status.get('warnings', [])
        if warnings:
            content.append("\n⚠️  Warnings:\n", style="yellow")
            for warning in warnings[:3]:
                content.append(f"  • {warning}\n", style="yellow")
        else:
            content.append("\n✓ All risk checks passed", style="green")
        
        return Panel(content)
    
    def render_footer(self) -> Panel:
        """Render footer with controls"""
        controls = Text()
        controls.append("Controls: ", style="bold")
        controls.append("[Q] Quit  [R] Refresh  [P] Pause  [B] Backtest  [S] Settings")
        
        return Panel(controls, style="dim")
    
    async def update_display(self, layout: Layout):
        """Update dashboard display"""
        # Get latest signals
        signals = await self.system.generate_signals()
        
        # Update layout sections
        layout["header"].update(self.render_header())
        layout["signals"].update(self.render_signals(signals))
        layout["portfolio"].update(self.render_portfolio())
        layout["metrics"].update(self.render_metrics())
        layout["risk"].update(self.render_risk())
        layout["footer"].update(self.render_footer())
        
        return signals
    
    async def run(self):
        """Run the dashboard"""
        layout = self.create_layout()
        
        with Live(layout, refresh_per_second=1, console=self.console) as live:
            while self.running:
                try:
                    # Update display
                    signals = await self.update_display(layout)
                    
                    # Execute signals if any
                    if signals:
                        await self.system.execute_signals(signals, paper_trade=True)
                    
                    # Monitor positions
                    await self.system.monitor_positions()
                    
                    # Wait before next update
                    await asyncio.sleep(5)
                    
                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    await asyncio.sleep(5)

class CLI:
    """Command-line interface for trading system"""
    
    def __init__(self):
        self.console = Console()
        self.system = PersonalTradingSystem()
        
    def run_backtest(self, args):
        """Run backtest with specified parameters"""
        self.console.print("\n[bold blue]Running Backtest...[/bold blue]\n")
        
        backtester = Backtester(initial_capital=args.capital)
        
        # Select strategy
        strategy_map = {
            'momentum': momentum_strategy,
            'mean_reversion': mean_reversion_strategy
        }
        
        strategy_func = strategy_map.get(args.strategy)
        if not strategy_func:
            self.console.print(f"[red]Unknown strategy: {args.strategy}[/red]")
            return
        
        # Run backtest
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Backtesting...", total=None)
            
            result = backtester.backtest_strategy(
                strategy_func,
                symbols=args.symbols.split(','),
                start_date=args.start,
                end_date=args.end
            )
            
            progress.stop()
        
        # Display results
        self.display_backtest_results(result)
        
        # Save results if requested
        if args.save:
            self.save_results(result, args.save)
    
    def display_backtest_results(self, result):
        """Display backtest results in a formatted table"""
        table = Table(title="Backtest Results", show_header=False)
        
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", justify="right")
        
        # Color code results
        def get_style(value, good_threshold, bad_threshold=None):
            if bad_threshold is not None:
                if value > good_threshold:
                    return "green"
                elif value < bad_threshold:
                    return "red"
                else:
                    return "yellow"
            else:
                return "green" if value > good_threshold else "red"
        
        metrics = [
            ("Total Return", f"{result.total_return:.2%}", 
             get_style(result.total_return, 0)),
            ("Sharpe Ratio", f"{result.sharpe_ratio:.2f}",
             get_style(result.sharpe_ratio, 1.0, 0.5)),
            ("Max Drawdown", f"{result.max_drawdown:.2%}",
             get_style(-result.max_drawdown, -0.1, -0.2)),
            ("Win Rate", f"{result.win_rate:.1%}",
             get_style(result.win_rate, 0.5, 0.4)),
            ("Total Trades", str(result.total_trades), "white"),
            ("Profit Factor", f"{result.profit_factor:.2f}",
             get_style(result.profit_factor, 1.5, 1.0)),
            ("Avg Win", f"${result.avg_win:.2f}", "green"),
            ("Avg Loss", f"${result.avg_loss:.2f}", "red"),
            ("Best Trade", f"{result.best_trade:.2%}", "green"),
            ("Worst Trade", f"{result.worst_trade:.2%}", "red"),
            ("Recovery Factor", f"{result.recovery_factor:.2f}",
             get_style(result.recovery_factor, 2.0, 1.0)),
            ("Calmar Ratio", f"{result.calmar_ratio:.2f}",
             get_style(result.calmar_ratio, 1.0, 0.5))
        ]
        
        for metric, value, style in metrics:
            table.add_row(metric, Text(value, style=style))
        
        self.console.print(table)
    
    def save_results(self, result, filename):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(result.__dict__, f, indent=2, default=str)
        self.console.print(f"\n[green]Results saved to {filename}[/green]")
    
    async def run_live(self, args):
        """Run live trading dashboard"""
        if args.config:
            # Load custom configuration
            with open(args.config, 'r') as f:
                config = json.load(f)
                self.system.config.update(config)
        
        dashboard = TradingDashboard(self.system)
        await dashboard.run()
    
    def run_analysis(self, args):
        """Run portfolio analysis"""
        self.console.print("\n[bold blue]Portfolio Analysis[/bold blue]\n")
        
        metrics = self.system.get_portfolio_metrics()
        
        # Display current metrics
        table = Table(title="Current Portfolio Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        for key, value in metrics.items():
            if key != 'strategy_performance':
                if isinstance(value, float):
                    table.add_row(key.replace('_', ' ').title(), f"{value:.2f}")
                else:
                    table.add_row(key.replace('_', ' ').title(), str(value))
        
        self.console.print(table)
        
        # Strategy performance breakdown
        if metrics.get('strategy_performance'):
            self.console.print("\n[bold]Strategy Performance:[/bold]")
            
            strat_table = Table()
            strat_table.add_column("Strategy", style="cyan")
            strat_table.add_column("Wins", justify="right")
            strat_table.add_column("Losses", justify="right")
            strat_table.add_column("Total Return", justify="right")
            
            for strategy, perf in metrics['strategy_performance'].items():
                strat_table.add_row(
                    strategy,
                    str(perf['wins']),
                    str(perf['losses']),
                    f"{perf['total_return']:.2%}"
                )
            
            self.console.print(strat_table)

def main():
    parser = argparse.ArgumentParser(description="Personal Trading System CLI")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live trading dashboard')
    live_parser.add_argument('--config', help='Configuration file path')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--strategy', required=True, 
                                help='Strategy to test (momentum, mean_reversion)')
    backtest_parser.add_argument('--symbols', required=True,
                                help='Comma-separated list of symbols')
    backtest_parser.add_argument('--start', required=True,
                                help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', required=True,
                                help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--capital', type=float, default=100000,
                                help='Starting capital')
    backtest_parser.add_argument('--save', help='Save results to file')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analyze', help='Analyze portfolio')
    
    args = parser.parse_args()
    
    cli = CLI()
    
    if args.command == 'live':
        asyncio.run(cli.run_live(args))
    elif args.command == 'backtest':
        cli.run_backtest(args)
    elif args.command == 'analyze':
        cli.run_analysis(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()