import pandas as pd
import numpy as np

def generate_full_latex_table(csv_path):
    """
    Parses a CSV file with detailed evaluation results and generates a
    multi-page, rotated LaTeX longtable, preserving the original structure
    but reordering metrics and sorting by model → prompt → temperature.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return f"Error: The file at {csv_path} was not found."
    except Exception as e:
        return f"An error occurred while reading the CSV file: {e}"

    # --- 0. Rename and drop unmapped prompts ---
    df.rename(columns={'Configuration': 'Model'}, inplace=True)
    prompt_mapping = {
        'current_user_template.txt': 'P1',
        'previous_user_template.txt': 'P2'
    }
    df = df[df['Prompt'].isin(prompt_mapping)]
    df['Prompt'] = df['Prompt'].map(prompt_mapping)

    # --- 1. Map models and sort ---
    model_name_mapping = {
        'GENAI_SHARED_VERTEXAI_GOOGLE_GEMINI_15_FLASH': 'Gemini 1.5 Flash',
        'GENAI_SHARED_VERTEXAI_GOOGLE_GEMINI_15_PRO': 'Gemini 1.5 Pro',
        'GENAI_SHARED_VERTEXAI_GOOGLE_GEMINI_2_FLASH': 'Gemini 2.0 Flash',
        'GENAI_SHARED_VERTEXAI_GOOGLE_GEMINI_2_FLASH_LITE': 'Gemini 2.0 Flash Lite',
        'GENAI_SHARED_VERTEXAI_GOOGLE_GEMINI_25_PRO': 'Gemini 2.5 Pro',
        'GENAI_SHARED_VERTEXAI_GOOGLE_GEMINI_25_FLASH': 'Gemini 2.5 Flash',
        'GENAI_SHARED_VERTEXAI_ANTHROPIC_CLAUDE_35_SONNET': 'Claude 3.5 Sonnet',
        'GENAI_SHARED_VERTEXAI_ANTHROPIC_CLAUDE_35_SONNET_V2': 'Claude 3.5 Sonnet v2',
        'GENAI_SHARED_VERTEXAI_ANTHROPIC_CLAUDE_37_SONNET': 'Claude 3.7 Sonnet',
        'GENAI_SHARED_VERTEXAI_ANTHROPIC_CLAUDE_4_SONNET': 'Claude 4.0 Sonnet',
        'GENAI_SHARED_BEDROCK_ANTHROPIC_CLAUDE_3_HAIKU': 'Claude 3 Haiku',
        'GENAI_SHARED_AZURE_OPENAI_GPT_4_OMNI': 'GPT-4 Omni',
        'GENAI_SHARED_AZURE_OPENAI_GPT_4_OMNI_2024_20_11': 'GPT-4 Omni (2024-11-20)',
        'GENAI_SHARED_AZURE_OPENAI_GPT_4_OMNI_MINI': 'GPT-4 Omni Mini',
        'GENAI_SHARED_AZURE_OPENAI_GPT_41_NANO': 'GPT-4.1 Nano',
        'GENAI_SHARED_AZURE_OPENAI_GPT_41': 'GPT-4.1',
        'GENAI_SHARED_AZURE_OPENAI_O1_MINI': 'O1 Mini',
        'GENAI_SHARED_AZURE_OPENAI_O1': 'O1',
        'GENAI_SHARED_AZURE_OPENAI_O3': 'O3',
        'GENAI_SHARED_AZURE_OPENAI_O3_MINI': 'O3 Mini',
        'GENAI_SHARED_AZURE_OPENAI_O4_MINI': 'O4 Mini',
        'GENAI_SHARED_AZURE_OPENAI_TEXT_ADA_002': 'Ada-002',
        'GENAI_SHARED_AZURE_OPENAI_TEXT_EMBEDDING_003_LARGE': 'Embedding-003 Large',
        'GENAI_SHARED_BEDROCK_AMAZON_TITAN_EMBED_TEXT_V1': 'Titan Embed v1',
        'GENAI_SHARED_VERTEXAI_GOOGLE_TEXTEMBEDDING_GECKO': 'Gecko'
    }
    # Create column with printable name, sort by model→prompt→temp
    df['MappedModel'] = df['Model'].map(model_name_mapping).fillna(df['Model'])
    df.sort_values(['MappedModel', 'Prompt', 'Temperature'], inplace=True)

    # --- 2. Metrics setup ---
    metric_order = {
        'Answer Relevancy': 'AR',
        'Correctness': 'C',
        'Faithfulness': 'F',
        'Hallucination': 'H',
        'Specific Information Accuracy': 'SIA'
    }
    metrics = list(metric_order)
    abbrevs = list(metric_order.values())

    avg_cols    = [f"{m} (Avg)"      for m in metrics]
    gpt4o_cols  = [f"{m} (GPT-4o)"    for m in metrics]
    claude_cols = [f"{m} (Sonnet3.5)" for m in metrics]

    df['Sum'] = df[avg_cols].sum(axis=1)

    # --- 3. Build LaTeX longtable ---
    header_abbr = " & ".join(f"\\textbf{{{a}}}" for a in abbrevs)
    latex = fr"""
% Requires in preamble:
%   \usepackage{{lscape,longtable,booktabs,adjustbox}}  % adjustbox only if you use method 2
\begin{{landscape}}
% ==== Method 1: smaller font & tighter columns ====
\small
\setlength{{\tabcolsep}}{{4pt}}       % default is ~6pt
\renewcommand{{\arraystretch}}{{0.9}}  % tighten row height
\setlength{{\LTleft}}{{0pt}}          % left align longtable
\setlength{{\LTright}}{{0pt}}         % right align longtable

% ==== Method 2 (optional): force to 90% of width ====
\begin{{adjustbox}}{{width=0.9\textwidth,center}}

\begin{{longtable}}{{|l|c|c|{'c'*5}|c|{'c'*5}|{'c'*5}|}}
\caption{{DeepEval Generation Quality Benchmark Results (Reordered)}} \\
\toprule
\textbf{{Model}} & \textbf{{Prompt}} & \textbf{{Temp}} & \multicolumn{{5}}{{c|}}{{\textbf{{Avg. Scores}}}} & \textbf{{Total}} & \multicolumn{{5}}{{c|}}{{\textbf{{GPT-4o}}}} & \multicolumn{{5}}{{c|}}{{\textbf{{Claude 3.5}}}} \\
 & & & {header_abbr} & & {header_abbr} & {header_abbr} \\
\midrule
\endfirsthead

\multicolumn{{{3+5+1+5+5}}}{{c}}{{\tablename\ \thetable{{}} -- continued from previous page}} \\
\midrule
\textbf{{Model}} & \textbf{{Prompt}} & \textbf{{Temp}} & {header_abbr} & & {header_abbr} & {header_abbr} \\
\midrule
\endhead

\midrule
\multicolumn{{{3+5+1+5+5}}}{{r}}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot
"""

    for _, r in df.iterrows():
        m = str(r['MappedModel']).replace('_', r'\_')
        p = r['Prompt']
        t = f"{r['Temperature']:.1f}" if pd.notna(r['Temperature']) else '-'
        avg_vals    = " & ".join(f"{r[c]:.2f}" for c in avg_cols)
        g4o_vals    = " & ".join(f"{r[c]:.2f}" for c in gpt4o_cols)
        claude_vals = " & ".join(f"{r[c]:.2f}" for c in claude_cols)
        total       = f"{r['Sum']:.2f}"
        latex += f"{m} & {p} & {t} & {avg_vals} & {total} & {g4o_vals} & {claude_vals} \\\\\n"

    latex += r"""\end{longtable}
\end{adjustbox}      % close Method 2
\end{landscape}
"""
    return latex

if __name__ == "__main__":
    csv_file_path = 'macro_evaluation_results.csv'
    output_filename = 'latex_table.txt'
    latex_output = generate_full_latex_table(csv_file_path)
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(latex_output)
        print(f"LaTeX table successfully saved to {output_filename}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")