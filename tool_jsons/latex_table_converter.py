import json

# Load the JSON file
names=[
    "Electronic_Topological_Descriptors",
    "Fragment_Based_Functional_Groups",
    "Identifiers_and_Representations",
    "Other_Descriptors",
    "Structural_Descriptors",
    ]

for name in names:
    with open(f"{name}.json", "r") as f:
        data = json.load(f)

    # Generate LaTeX table rows
    latex_rows = ""
    for item in data:
        function_name = item["function"]["name"]
        function_name = function_name.replace("_","\_")
        description = item["function"]["description"].strip().replace("\n", " ").replace("  ", " ")
        latex_rows += f"\\texttt{{{function_name}}} & {description} \\\\\n\\hline\n"

    # Wrap in LaTeX table format
    latex_table = f"""
    \\begin{{longtable}}{{|l|p{{0.76\\textwidth}}|}}
    \\hline
    \\textbf{{Function}} & \\textbf{{Description}} \\\\
    \\hline
    {latex_rows}
    \\caption{{List of {name.replace("_", " ")} tools}}
    \\label{{tab:{name}}}
    \\end{{longtable}}
    """

    # Save to file
    output_path = f"{name}_table.txt"
    with open(output_path, "w") as out_file:
        out_file.write(latex_table)

    print(f"LaTeX table written to {output_path}")
