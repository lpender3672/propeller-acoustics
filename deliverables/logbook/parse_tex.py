import re

def escape_latex(s):
    """
    Escapes LaTeX special characters in a string.
    """
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    pattern = re.compile('|'.join(re.escape(key) for key in replacements.keys()))
    return pattern.sub(lambda match: replacements[match.group()], s)

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
            current['date'], current['author'], current['message'] = parts
        elif line and line[0].isdigit():
            parts = line.split('\t')
            if len(parts) == 3:
                added, removed, filename = parts
                try:
                    current['files'].append((int(added), int(removed), filename))
                except ValueError:
                    continue
    if current:
        entries.append(current)

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
            f.write(r"\section*{" + f"{escape_latex(entry['date'])} --- {escape_latex(entry['author'])}" + "}\n")
            f.write(r"\textbf{Message:} " + escape_latex(entry['message']) + "\n\n")
            if entry['files']:
                f.write(r"\begin{longtable}{|p{10cm}|r|r|}" + "\n")
                f.write(r"\hline \textbf{File} & \textbf{+Lines} & \textbf{-Lines} \\" + "\n\\hline\n")
                for added, removed, fname in entry['files']:
                    f.write(f"{escape_latex(fname)} & {added} & {removed} \\\\\n\\hline\n")
                f.write(r"\end{longtable}" + "\n\n")

        f.write(r"\end{document}")

if __name__ == "__main__":

    # first run the command
    # git log --date=short --pretty=format:"--COMMIT--%n%ad | %an | %s%n" --numstat > deliverables/logbook/detailed_commits.txt

    input_file = 'deliverables/logbook/detailed_commits.txt'  # Replace with your actual git log file
    output_file = 'deliverables/logbook/logbook.tex'
    parse_git_log_with_numstat(input_file, output_file)
    print(f"Logbook written to {output_file}")

