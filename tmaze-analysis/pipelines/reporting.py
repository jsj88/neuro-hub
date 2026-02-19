"""
Automated report generation for T-maze analysis.

Creates HTML reports with figures, statistics tables, and
publication-ready outputs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import json

try:
    from jinja2 import Template
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = 'T-Maze Analysis Report'
    author: str = ''
    include_figures: bool = True
    include_tables: bool = True
    include_stats: bool = True
    theme: str = 'default'
    output_format: str = 'html'


def generate_report(
    results: Dict[str, Any],
    figures: Optional[Dict[str, Path]] = None,
    output_path: Path = None,
    config: Optional[ReportConfig] = None
) -> Path:
    """
    Generate HTML analysis report.

    Parameters
    ----------
    results : Dict
        Analysis results (classification, RSA, etc.)
    figures : Dict[str, Path], optional
        Paths to figure files
    output_path : Path
        Output file path
    config : ReportConfig, optional
        Report configuration

    Returns
    -------
    Path
        Path to generated report
    """
    if config is None:
        config = ReportConfig()

    if output_path is None:
        output_path = Path('tmaze_report.html')
    output_path = Path(output_path)

    # Build report sections
    sections = []

    # Summary section
    sections.append(_build_summary_section(results))

    # Classification results
    if 'classification' in results:
        sections.append(_build_classification_section(results['classification']))

    # RSA results
    if 'rsa' in results:
        sections.append(_build_rsa_section(results['rsa']))

    # Connectivity results
    if 'connectivity' in results:
        sections.append(_build_connectivity_section(results['connectivity']))

    # Group statistics
    if 'group_stats' in results:
        sections.append(_build_stats_section(results['group_stats']))

    # Figures
    if figures and config.include_figures:
        sections.append(_build_figures_section(figures))

    # Generate HTML
    html = _render_html(sections, config, figures)

    # Write output
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Report generated: {output_path}")
    return output_path


def _build_summary_section(results: Dict) -> Dict:
    """Build summary section."""
    summary = {
        'title': 'Analysis Summary',
        'type': 'summary',
        'content': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'n_subjects': results.get('n_subjects', 'N/A'),
            'analyses_run': list(results.keys())
        }
    }
    return summary


def _build_classification_section(results: Dict) -> Dict:
    """Build classification results section."""
    return {
        'title': 'Classification Results',
        'type': 'classification',
        'content': results
    }


def _build_rsa_section(results: Dict) -> Dict:
    """Build RSA results section."""
    return {
        'title': 'Representational Similarity Analysis',
        'type': 'rsa',
        'content': results
    }


def _build_connectivity_section(results: Dict) -> Dict:
    """Build connectivity results section."""
    return {
        'title': 'Connectivity Analysis',
        'type': 'connectivity',
        'content': results
    }


def _build_stats_section(results: Dict) -> Dict:
    """Build statistics section."""
    return {
        'title': 'Group Statistics',
        'type': 'statistics',
        'content': results
    }


def _build_figures_section(figures: Dict[str, Path]) -> Dict:
    """Build figures section."""
    return {
        'title': 'Figures',
        'type': 'figures',
        'content': figures
    }


def _render_html(
    sections: List[Dict],
    config: ReportConfig,
    figures: Optional[Dict[str, Path]]
) -> str:
    """Render HTML from sections."""
    template = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 30px;
            margin: -20px -20px 30px -20px;
        }
        .header h1 { margin: 0; }
        .section {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th { background: #3498db; color: white; }
        tr:hover { background: #f5f5f5; }
        .figure {
            text-align: center;
            margin: 20px 0;
        }
        .figure img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .figure-caption {
            color: #666;
            font-style: italic;
            margin-top: 10px;
        }
        .stat-box {
            display: inline-block;
            background: #ecf0f1;
            padding: 15px 25px;
            margin: 5px;
            border-radius: 5px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 12px;
        }
        .significant { color: #27ae60; font-weight: bold; }
        .not-significant { color: #95a5a6; }
        .footer {
            text-align: center;
            color: #7f8c8d;
            padding: 20px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        {% if author %}<p>{{ author }}</p>{% endif %}
        <p>Generated: {{ date }}</p>
    </div>

    {% for section in sections %}
    <div class="section">
        <h2>{{ section.title }}</h2>
        {% if section.type == 'summary' %}
            <div class="stat-box">
                <div class="stat-value">{{ section.content.n_subjects }}</div>
                <div class="stat-label">Subjects</div>
            </div>
            <p>Analyses: {{ section.content.analyses_run | join(', ') }}</p>
        {% elif section.type == 'classification' %}
            {% if section.content.accuracy is defined %}
            <div class="stat-box">
                <div class="stat-value">{{ "%.1f" | format(section.content.accuracy * 100) }}%</div>
                <div class="stat-label">Accuracy</div>
            </div>
            {% endif %}
            {% if section.content.auc is defined %}
            <div class="stat-box">
                <div class="stat-value">{{ "%.3f" | format(section.content.auc) }}</div>
                <div class="stat-label">AUC</div>
            </div>
            {% endif %}
        {% elif section.type == 'statistics' %}
            <table>
                <tr><th>Test</th><th>Statistic</th><th>p-value</th><th>Effect Size</th></tr>
                {% for name, stat in section.content.items() %}
                <tr>
                    <td>{{ name }}</td>
                    <td>{{ "%.3f" | format(stat.statistic) if stat.statistic else "N/A" }}</td>
                    <td class="{% if stat.p_value < 0.05 %}significant{% else %}not-significant{% endif %}">
                        {{ "%.4f" | format(stat.p_value) if stat.p_value else "N/A" }}
                        {% if stat.p_value < 0.001 %}***{% elif stat.p_value < 0.01 %}**{% elif stat.p_value < 0.05 %}*{% endif %}
                    </td>
                    <td>{{ "%.3f" | format(stat.effect_size) if stat.effect_size else "N/A" }}</td>
                </tr>
                {% endfor %}
            </table>
        {% elif section.type == 'figures' %}
            {% for name, path in section.content.items() %}
            <div class="figure">
                <img src="{{ path }}" alt="{{ name }}">
                <div class="figure-caption">{{ name }}</div>
            </div>
            {% endfor %}
        {% endif %}
    </div>
    {% endfor %}

    <div class="footer">
        Generated by tmaze-analysis toolkit
    </div>
</body>
</html>'''

    if HAS_JINJA:
        tmpl = Template(template)
        return tmpl.render(
            title=config.title,
            author=config.author,
            date=datetime.now().strftime('%Y-%m-%d %H:%M'),
            sections=sections
        )
    else:
        # Simple fallback without Jinja
        return _render_simple_html(sections, config)


def _render_simple_html(sections: List[Dict], config: ReportConfig) -> str:
    """Simple HTML rendering without Jinja."""
    html = f'''<!DOCTYPE html>
<html>
<head><title>{config.title}</title></head>
<body>
<h1>{config.title}</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
'''
    for section in sections:
        html += f"<h2>{section['title']}</h2>\n"
        html += f"<pre>{json.dumps(section['content'], indent=2, default=str)}</pre>\n"

    html += '</body></html>'
    return html


def create_figure_gallery(
    figure_dir: Path,
    output_path: Path,
    title: str = 'Figure Gallery'
) -> Path:
    """
    Create HTML gallery of all figures in a directory.

    Parameters
    ----------
    figure_dir : Path
        Directory containing figures
    output_path : Path
        Output HTML file path
    title : str
        Gallery title

    Returns
    -------
    Path
        Path to gallery file
    """
    figure_dir = Path(figure_dir)
    output_path = Path(output_path)

    # Find all image files
    extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
    figures = []
    for ext in extensions:
        figures.extend(figure_dir.glob(f'*{ext}'))

    figures = sorted(figures)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }}
        .figure {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
        .figure img {{ max-width: 100%; height: auto; }}
        .caption {{ margin-top: 10px; color: #666; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <div class="gallery">
'''

    for fig in figures:
        name = fig.stem.replace('_', ' ').title()
        html += f'''
        <div class="figure">
            <img src="{fig.name}" alt="{name}">
            <div class="caption">{name}</div>
        </div>
'''

    html += '''
    </div>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def statistics_table(
    stats: Dict[str, Any],
    format: str = 'markdown'
) -> str:
    """
    Create formatted statistics table.

    Parameters
    ----------
    stats : Dict
        Statistics results
    format : str
        'markdown', 'latex', or 'html'

    Returns
    -------
    str
        Formatted table
    """
    if format == 'markdown':
        lines = ['| Metric | Value | CI 95% | p-value |', '|--------|-------|--------|---------|']

        for name, value in stats.items():
            if isinstance(value, dict):
                val = value.get('mean', value.get('statistic', 'N/A'))
                ci = f"[{value.get('ci_lower', 'N/A')}, {value.get('ci_upper', 'N/A')}]"
                p = value.get('p_value', 'N/A')
                if isinstance(p, float):
                    p = f"{p:.4f}" + ('*' if p < 0.05 else '')
            else:
                val = value
                ci = 'N/A'
                p = 'N/A'

            if isinstance(val, float):
                val = f"{val:.4f}"

            lines.append(f"| {name} | {val} | {ci} | {p} |")

        return '\n'.join(lines)

    elif format == 'latex':
        lines = [
            r'\begin{table}[h]',
            r'\centering',
            r'\begin{tabular}{lccc}',
            r'\hline',
            r'Metric & Value & 95\% CI & p-value \\',
            r'\hline'
        ]

        for name, value in stats.items():
            if isinstance(value, dict):
                val = value.get('mean', value.get('statistic', 'N/A'))
                ci = f"[{value.get('ci_lower', 'N/A')}, {value.get('ci_upper', 'N/A')}]"
                p = value.get('p_value', 'N/A')
                if isinstance(p, float):
                    if p < 0.001:
                        p = r'$<$.001***'
                    elif p < 0.01:
                        p = f'{p:.3f}**'
                    elif p < 0.05:
                        p = f'{p:.3f}*'
                    else:
                        p = f'{p:.3f}'
            else:
                val = value
                ci = '--'
                p = '--'

            if isinstance(val, float):
                val = f"{val:.3f}"

            name = name.replace('_', r'\_')
            lines.append(f'{name} & {val} & {ci} & {p} \\\\')

        lines.extend([
            r'\hline',
            r'\end{tabular}',
            r'\caption{Statistical results}',
            r'\label{tab:stats}',
            r'\end{table}'
        ])

        return '\n'.join(lines)

    elif format == 'html':
        return _render_simple_html([{
            'title': 'Statistics',
            'type': 'statistics',
            'content': stats
        }], ReportConfig())

    else:
        raise ValueError(f"Unknown format: {format}")


def export_for_publication(
    results: Dict[str, Any],
    output_dir: Path,
    format: str = 'latex'
) -> Dict[str, Path]:
    """
    Export results in publication-ready format.

    Parameters
    ----------
    results : Dict
        Analysis results
    output_dir : Path
        Output directory
    format : str
        'latex' or 'word'

    Returns
    -------
    Dict[str, Path]
        Paths to exported files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    # Export main statistics table
    if 'group_stats' in results:
        table = statistics_table(results['group_stats'], format='latex' if format == 'latex' else 'markdown')
        table_file = output_dir / f'table_statistics.{"tex" if format == "latex" else "md"}'
        with open(table_file, 'w') as f:
            f.write(table)
        exported['statistics_table'] = table_file

    # Export summary JSON
    summary_file = output_dir / 'results_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    exported['summary_json'] = summary_file

    print(f"Exported {len(exported)} files to: {output_dir}")
    return exported
