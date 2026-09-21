# Computational Imaging Lab 3

## Report Organization

**Report file:** `lab3_report.pdf` (single consolidated report)

Do NOT create separate PDFs for each task that begins with "D" (e.g., D1, D2, etc.). Instead, write all sections to the same LaTeX document `lab3_report.tex` and compile to the unified `lab3_report.pdf`.

Structure:
- Each task gets a `\section*{}` in the same document
- Update and recompile lab3_report.tex incrementally

## Visualization Standards

### Grayscale Colormaps

**Inverted grayscale**: Use `'gray_r'` (reversed grayscale) for standard image visualization. In this colormap:
- Dark pixels represent large values
- Light pixels represent small values
- This follows the convention where high intensity = darkness

### Difference Image Colormaps

For difference images (e.g., `Ax - A^T x`), use a custom colormap that:
- Maps 0 (zero difference) to gray (0.5 on the original [0, 1] scale)
- Uses diverging colors: negative differences in cool tones, positive in warm tones
- Centers at zero for intuitive interpretation of symmetry

Implementation: Use `matplotlib.colors.TwoSlopeNorm` or a custom diverging colormap centered at the middle value.
