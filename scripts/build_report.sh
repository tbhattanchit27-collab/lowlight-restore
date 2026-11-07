#!/bin/sh
# Builds report/Project_Report_P1.docx and .pdf from the markdown docs.
# Needs pandoc (and LibreOffice for the PDF). Edit the names below first.
TEAM="Team APex decs: Anish Gowda P (1RUA24CSE0053), Anchit Tushar Bhatt (1RUA24CSE0049), Amith S (1RUA24CSE0038), Chethan Chouhan (1RUA24CSE0104)"
cd "$(dirname "$0")/.." && mkdir -p report
{
  echo "% Robust Low-Light Image Restoration and Enhancement (P1)"
  echo "% $TEAM"
  echo "% Computer Vision Hackathon 2026"
  echo
  sed '1,4d' DESIGN.md
  echo; printf '%s\n' '```{=openxml}' '<w:p><w:r><w:br w:type="page"/></w:r></w:p>' '```'; echo
  sed '1,2d' eval_P1_TXX.md | sed 's/^# Evaluation/# Evaluation Results/'
  echo
  sed '1d' ABLATION_AND_FAILURES.md | sed '1i # Ablation Study and Failure Analysis'
  echo
  echo "# Appendix: example results"; echo
  echo '![Real dark photos: input, classical, Zero-DCE, hybrid](docs/compare_real.jpg)'; echo
  echo '![Synthetic test: input, classical, Zero-DCE, hybrid, ground truth](docs/compare_synthetic.jpg)'; echo
  echo '![PSNR and SSIM on the official test sets](docs/results_chart_lol.png)'; echo
  echo '![Intermediate steps of the hybrid pipeline](docs/intermediate.jpg)'
} > report/report.md
pandoc report/report.md -o report/Project_Report_P1.docx --resource-path=.
(cd report && soffice --headless --convert-to pdf Project_Report_P1.docx >/dev/null 2>&1) || echo "LibreOffice not found: open the .docx and save as PDF"
echo "report/Project_Report_P1.docx written"
