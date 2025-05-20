# converts images to pdfs for faster report generation

from PIL import Image
from pathlib import Path


def convert_folder_of_images_to_pdfs(input_dir, output_dir = None):

    if output_dir is None:
        output_dir = input_dir

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for img_path in input_dir.glob("*.[jp][pn]g"):
        img = Image.open(img_path).convert("RGB")
        output_path = output_dir / (img_path.stem + ".pdf")
        img.save(output_path)

if __name__ == "__main__":

    input_dir = 'deliverables/final_report/figures'
    convert_folder_of_images_to_pdfs(input_dir)
    input_dir = 'deliverables/practical_summary/figures'
    output_dir = 'deliverables/final_report/figures'
    convert_folder_of_images_to_pdfs(input_dir, output_dir)
    print("Done.")