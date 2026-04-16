"""
FPV Lap Counter
Анализирует видео FPV-полётов, считает времена кругов,
находит лучшие 3 подряд.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Optional

import typer
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich import print as rprint
from rich.panel import Panel

app = typer.Typer(help="FPV Lap Counter — анализ времён кругов из видео")
console = Console()


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


@app.command()
def analyze(
    videos: List[Path] = typer.Argument(..., help="Видеофайлы для анализа"),
    min_lap: float = typer.Option(3.0, "--min-lap", help="Минимальное время круга (секунды)"),
    peak_height: float = typer.Option(0.08, "--peak-height", help="Минимальная чувствительность детектора ворот (0..1)"),
    peak_prominence: float = typer.Option(0.04, "--prominence", help="Минимальная выраженность пика"),
    sample_every: int = typer.Option(3, "--sample", help="Анализировать каждый N-й кадр (скорость vs точность)"),
    no_ocr: bool = typer.Option(False, "--no-ocr", help="Отключить OCR таймера (использовать только frame timestamp)"),
    no_gpu: bool = typer.Option(False, "--no-gpu", help="Отключить GPU для OCR"),
    debug_frames: bool = typer.Option(False, "--debug-frames", help="Сохранить кадры каждого пролёта"),
    output_json: Optional[Path] = typer.Option(None, "--json", help="Сохранить результаты в JSON"),
    lm_studio: bool = typer.Option(False, "--lm-studio", help="Использовать LM Studio для верификации"),
    lm_url: str = typer.Option("http://localhost:1234/v1", "--lm-url", help="URL LM Studio API"),
    best_n: int = typer.Option(3, "--best-n", help="Найти лучшие N подряд кругов"),
):
    """
    Анализирует FPV видео и считает времена кругов.

    Пример:
        python main.py race1.mp4 race2.mp4 --best-n 3
    """
    from src.gate_detector import GateDetector
    from src.lap_analyzer import analyze_video, compare_analyses
    from src.osd_reader import OSDReader

    # Валидация входных файлов
    valid_videos = []
    for v in videos:
        if not v.exists():
            console.print(f"[red]Файл не найден: {v}[/red]")
        else:
            valid_videos.append(v)

    if not valid_videos:
        console.print("[red]Нет доступных видеофайлов.[/red]")
        raise typer.Exit(1)

    # Инициализация OCR
    osd_reader = None
    if not no_ocr:
        with console.status("[bold]Загрузка EasyOCR модели...[/bold]"):
            try:
                osd_reader = OSDReader(use_gpu=not no_gpu, verbose=False)
                console.print("[green]✓ EasyOCR готов[/green]")
            except ImportError:
                console.print("[yellow]⚠ EasyOCR не установлен, OCR отключён[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠ OCR ошибка: {e}[/yellow]")

    # Инициализация LM Studio (опционально)
    lm_client = None
    if lm_studio:
        with console.status("[bold]Подключение к LM Studio...[/bold]"):
            try:
                from src.lm_studio_client import LMStudioClient
                lm_client = LMStudioClient(base_url=lm_url, verbose=True)
                if lm_client.is_available():
                    console.print(f"[green]✓ LM Studio подключён: {lm_client.model}[/green]")
                else:
                    console.print("[yellow]⚠ LM Studio недоступен[/yellow]")
                    lm_client = None
            except Exception as e:
                console.print(f"[yellow]⚠ LM Studio ошибка: {e}[/yellow]")

    detector = GateDetector(
        min_lap_sec=min_lap,
        peak_height=peak_height,
        peak_prominence=peak_prominence,
        sample_every_n=sample_every,
        verbose=False,
    )

    all_analyses = []

    for video_path in valid_videos:
        console.rule(f"[bold blue]{video_path.name}[/bold blue]")

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Анализ видео...", total=None)

            def update_progress(current, total):
                pct = int(current / total * 100)
                progress.update(task, description=f"Анализ видео... {pct}%")

            try:
                gate_passes, scores = detector.process_video(
                    str(video_path),
                    osd_reader=osd_reader,
                    progress_callback=update_progress,
                )
                progress.update(task, description="Анализ завершён ✓")
            except Exception as e:
                console.print(f"[red]Ошибка обработки {video_path.name}: {e}[/red]")
                continue

        if not gate_passes:
            console.print("[yellow]⚠ Пролётов через ворота не обнаружено.[/yellow]")
            console.print(
                "[dim]Попробуйте уменьшить --peak-height (напр. 0.04) "
                "или --prominence (напр. 0.02)[/dim]"
            )
            continue

        console.print(f"Обнаружено пролётов: [bold]{len(gate_passes)}[/bold]")

        analysis = analyze_video(
            str(video_path),
            gate_passes,
            use_osd=(osd_reader is not None),
            max_consecutive=best_n,
        )
        all_analyses.append(analysis)

        # Таблица кругов
        if analysis.laps:
            table = Table(title="Времена кругов", show_lines=True)
            table.add_column("Круг", style="cyan", width=6)
            table.add_column("Время", style="white", width=10)
            table.add_column("Старт (с)", style="dim", width=10)
            table.add_column("Оценка", style="dim", width=8)

            for lap in analysis.laps:
                is_best = (analysis.best_lap and lap.lap_number == analysis.best_lap.lap_number)
                style = "bold green" if is_best else ""
                marker = " ★" if is_best else ""
                table.add_row(
                    str(lap.lap_number),
                    lap.duration_str + marker,
                    f"{lap.start_pass.timestamp_sec:.1f}",
                    f"{lap.start_pass.gate_score:.3f}",
                    style=style,
                )

            console.print(table)

        # Лучший круг
        if analysis.best_lap:
            console.print(
                f"\n[bold green]★ Лучший круг:[/bold green] "
                f"#{analysis.best_lap.lap_number} — {analysis.best_lap.duration_str}"
            )

        # Лучшие N подряд
        best = analysis.best_n_consecutive.get(best_n)
        if best:
            lap_nums = [str(l.lap_number) for l in best.laps]
            console.print(
                Panel(
                    f"Круги: [bold]{', '.join(lap_nums)}[/bold]\n"
                    f"Суммарно: [bold yellow]{best.total_str}[/bold yellow]\n"
                    f"Среднее:  [bold cyan]{best.average_str}[/bold cyan]",
                    title=f"[bold]★ Лучшие {best_n} подряд[/bold]",
                    border_style="yellow",
                )
            )

        # Отладочные кадры
        if debug_frames:
            output_dir = f"output/{video_path.stem}"
            detector.save_debug_frames(str(video_path), gate_passes, output_dir)

    # Сравнение нескольких видео
    if len(all_analyses) > 1:
        console.rule("[bold]Сравнение видео[/bold]")
        from src.lap_analyzer import compare_analyses as compare
        comparison = compare(all_analyses)

        if comparison.get("global_best_lap"):
            path, lap = comparison["global_best_lap"]
            video_name = Path(path).name
            console.print(
                f"\n[bold green]★ Лучший круг из всех:[/bold green] "
                f"{video_name} — круг #{lap.lap_number} — {lap.duration_str}"
            )

        if comparison.get("global_best_3_consecutive"):
            path, best3 = comparison["global_best_3_consecutive"]
            video_name = Path(path).name
            console.print(
                f"\n[bold yellow]★ Лучшие 3 подряд из всех:[/bold yellow] "
                f"{video_name} — {best3.total_str} (среднее {best3.average_str})"
            )

        # Рейтинг
        rankings_table = Table(title="Рейтинг видео", show_lines=True)
        rankings_table.add_column("Место", style="cyan", width=6)
        rankings_table.add_column("Видео", style="white")
        rankings_table.add_column(f"Лучший круг", style="green", width=10)
        rankings_table.add_column(f"Лучшие {best_n} подряд", style="yellow", width=12)
        rankings_table.add_column("Кол-во кругов", style="dim", width=13)

        for i, analysis in enumerate(comparison.get("rankings", []), 1):
            best_lap_str = analysis.best_lap.duration_str if analysis.best_lap else "—"
            best_consec = analysis.best_n_consecutive.get(best_n)
            best_consec_str = best_consec.total_str if best_consec else "—"
            rankings_table.add_row(
                str(i),
                Path(analysis.video_path).name,
                best_lap_str,
                best_consec_str,
                str(len(analysis.laps)),
            )
        console.print(rankings_table)

    # JSON экспорт
    if output_json and all_analyses:
        result = []
        for analysis in all_analyses:
            result.append({
                "video": str(Path(analysis.video_path).name),
                "gate_passes": len(analysis.gate_passes),
                "laps": [
                    {
                        "lap": lap.lap_number,
                        "duration_sec": round(lap.duration_sec, 3),
                        "duration_str": lap.duration_str,
                        "start_sec": round(lap.start_pass.timestamp_sec, 2),
                    }
                    for lap in analysis.laps
                ],
                "best_lap": {
                    "lap": analysis.best_lap.lap_number,
                    "duration_str": analysis.best_lap.duration_str,
                } if analysis.best_lap else None,
                "best_3_consecutive": {
                    "laps": [l.lap_number for l in analysis.best_3_consecutive.laps],
                    "total_str": analysis.best_3_consecutive.total_str,
                    "average_str": analysis.best_3_consecutive.average_str,
                } if analysis.best_3_consecutive else None,
            })

        output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        console.print(f"\n[green]✓ Результаты сохранены: {output_json}[/green]")


@app.command()
def calibrate(
    video: Path = typer.Argument(..., help="Видеофайл для калибровки"),
    output_dir: Path = typer.Option(Path("output/calibrate"), "--out", help="Куда сохранить кадры"),
    every_sec: float = typer.Option(1.0, "--every", help="Каждые N секунд брать кадр"),
):
    """
    Режим калибровки: сохраняет кадры с визуализацией детекции LED.
    Используй для подбора параметров --peak-height и --prominence.
    """
    import cv2
    from src.gate_detector import compute_gate_score, compute_led_mask

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * every_sec)

    scores = []
    saved = 0

    with console.status(f"Калибровка {video.name}..."):
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                score = compute_gate_score(frame)
                scores.append((frame_idx, frame_idx / fps, score))

                # Сохраняем с overlay
                led_mask = compute_led_mask(frame)
                h, w = frame.shape[:2]
                overlay = frame.copy()
                overlay[led_mask > 0] = [0, 255, 0]
                debug = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                cv2.putText(debug, f"t={frame_idx/fps:.1f}s score={score:.4f}",
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                out_path = output_dir / f"frame_{frame_idx:06d}_score{score:.4f}.jpg"
                cv2.imwrite(str(out_path), debug)
                saved += 1

            frame_idx += 1

    cap.release()

    # Показываем таблицу с топ scores
    console.print(f"\nСохранено {saved} кадров в {output_dir}/")

    table = Table(title="Top-20 кадров по gate score")
    table.add_column("Время (с)")
    table.add_column("Gate Score")
    table.add_column("Оценка")

    top = sorted(scores, key=lambda x: x[2], reverse=True)[:20]
    for frame_idx, t, score in top:
        bar = "█" * int(score * 30)
        table.add_row(f"{t:.1f}", f"{score:.4f}", bar)

    console.print(table)
    console.print("\n[dim]Файлы с высоким score = кадры около ворот.[/dim]")
    console.print("[dim]Используй это чтобы подобрать --peak-height[/dim]")


if __name__ == "__main__":
    app()
