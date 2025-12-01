#!/usr/bin/env python3
"""
Markdown to Typst Converter

This script converts a Markdown file to Typst syntax by:
1. Replacing all occurrences of '#' with '='
2. Removing all occurrences of '*'
3. Converting image references from Markdown to Typst format
"""

import re
import argparse
import os
import sys


def convert_markdown_to_typst(input_file, output_file=None):
    """
    Convert a Markdown file to Typst syntax.
    
    Args:
        input_file (str): Path to the input Markdown file
        output_file (str, optional): Path to the output Typst file
    """
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except IOError as e:
        print(f"Error reading input file: {e}")
        return False
    
    # Replace # with = (header syntax)
    content = content.replace('#', '=')
    
    # Process asterisks line by line
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        if line.startswith('='):
            # Remove all * from header lines
            line = line.replace('*', '')
        else:
            # Replace ** with * in non-header lines
            line = line.replace('**', '*')
        processed_lines.append(line)
    
    content = '\n'.join(processed_lines)
    
    # Remove all ➤ characters
    content = content.replace('➤ ', '')
    
    # Remove all • characters
    content = content.replace('• ', '')
    
    # Replace image references
    # Pattern to match ![](_some_image_path)
    image_pattern = r'!\[\]\(([^)]+)\)'
    # Replace with #figure(image("images/\1"))
    content = re.sub(image_pattern, r'#figure(image("images/\1"))', content)
    
    # Determine output file if not provided
    if output_file is None:
        base_name, _ = os.path.splitext(input_file)
        output_file = f"{base_name}_converted.typ"
    
    try:
        # Write the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
    except IOError as e:
        print(f"Error writing output file: {e}")
        return False
    
    print(f"Successfully converted {input_file} to {output_file}")
    return True


def main():
    """Main function to handle command-line arguments and execute the conversion."""
    parser = argparse.ArgumentParser(
        description='Convert Markdown to Typst syntax',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python md_to_typst.py input.md
  python md_to_typst.py input.md -o output.typ
        """
    )
    parser.add_argument('input_file', help='Input Markdown file')
    parser.add_argument('-o', '--output', help='Output Typst file (optional)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return 1
    
    # Convert the file
    if not convert_markdown_to_typst(args.input_file, args.output):
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())