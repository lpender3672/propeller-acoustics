import re

def parse_git_log_with_numstat(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    entries = []
    current = {}

    for line in lines:
        line = line.strip()
        if line.startswith('--COMMIT--'):
            if current:
                entries.append(current)
            current = {'files': []}
        elif '|' in line and 'files' in current:
            parts = [x.strip() for x in line.split('|')]
            if len(parts) == 3:
                current['date'], current['author'], current['message'] = parts
            else:
                print(f"Skipping malformed commit line: {line}")
        elif line and current.get('files') is not None and line[0].isdigit() or line.startswith('-'):
            parts = line.split('\t')
            if len(parts) == 3:
                added, removed, filename = parts
                try:
                    added = int(added) if added != '-' else 0
                    removed = int(removed) if removed != '-' else 0
                    current['files'].append((added, removed, filename))
                except ValueError:
                    print(f"Skipping non-numeric line: {line}")
    if current:
        entries.append(current)

    # Write LaTeX output
    with open(output_file, 'w') as f:
        f.write(r"""\documentclass{article}
\usepackage{longtable}
\usepackage{geometry}
\geometry{margin=1in}
\title{Project Logbook}
\date{}
\begin{document}
\maketitle

""")
        for entry in entries:
            if 'date' not in entry or 'author' not in entry or 'message' not in entry:
                continue
            f.write(r"\section*{" + f"{escape_latex(entry['date'])} --- {escape_latex(entry['author'])}" + "}\n")
            f.write(r"\textbf{Message:} " + escape_latex(entry['message']) + "\n\n")
            if entry['files']:
                f.write(r"\begin{longtable}{|p{10cm}|r|r|}" + "\n")
                f.write(r"\hline \textbf{File} & \textbf{+Lines} & \textbf{-Lines} \\" + "\n\\hline\n")
                for added, removed, fname in entry['files']:
                    f.write(f"{escape_latex(fname)} & {added} & {removed} \\\\\n\\hline\n")
                f.write(r"\end{longtable}" + "\n\n")

        f.write(r"\end{document}")