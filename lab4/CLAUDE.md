# Computational Imaging Lab 4

## General Code Writing

1. Every function has the exact signature given here, and a docstring saying what it does, what its inputs are, and what it returns.
2. Everything in float64, so that the cost differences you plot are not lost in rounding. 
3. No for loop over pixels.
4. Do not enforce a nonnegativity constraint on $x$. Leave $x$ unconstrained.
5. Fix a random seed for the noise and report the seed.

---

## Parameters for Every Experiment

1. Every reconstruction in this lab uses the same image and blur. 
2. The image is `kodim23.pgm` on $[0, 1]$
3. We use convolution with Levin kernel 1 (levin09_kernels/levin09_kernel1.txt) and circular boundaries.
4. The potential always uses $p = 2.0$ and $q = 1.2$.
5. $x$ is unconstrained.

### Noise Generation

1. Draw a noise image \(w\) with standard
      deviation \( \sigma_w \), add it to the blurred signal to form the data
      \( y = A x_{\text{true}} + w \), and reconstruct from that \(y\).
2. The symbol \( \sigma_w \) does two separate jobs,
      set to one value in this lab: it is the standard deviation of the noise
      you add, and it is the noise level the cost assumes in its data term
      \( \frac{1}{2\sigma_w^2}\|y - Ax\|^2 \). In general these two need not be
      equal.
3. Fix and
      report the random seed.

### Parameters for each Deliverable

  --------- --------------------------------------- -------------------- ------------ ----------------------------- ------------------------
  Item      Experiment                              Noise $\sigma_{w}$   $T$          $\sigma_{x}$                  Iterations
  D1        Potential & influence plots (Task 2a)   none \*              1            0.2 \*                        plot only
  D3        Surrogate plot (Task 4b)                none \*              1            0.2 \*                        plot only
  D5        Cost decreases each step (Task 6a)      0.02                 1            0.020                         $N = 50$, $\omega = 1$
  D6--D10   Reconstruction runs R1--R6 (Step 7)     0.02                 1 or 100 †   see the run table in Step 7   $N = 50$, $\omega = 1$
  --------- --------------------------------------- -------------------- ------------ ----------------------------- ------------------------

---


## Report Organization

**Report file:** `lab4_report.pdf` (single consolidated report)

Do NOT create separate PDFs for each task that begins with "D" (e.g., D1, D2, etc.). Instead, write all sections to the same markdown document `lab4_report.md` and compile to the unified `lab4_report.pdf`.

Each task has their own section.

## 2. Equations in Markdown Go in LaTeX, With Their Notation

**Rule**: In every `.md` file in this project, write equations as LaTeX —
`$...$` inline, `$$...$$` displayed — never as monospace/code-fence ASCII.
Immediately after the equation, define every symbol in it, then say what it
means physically. Reserve backticks for things that are literally code:
filenames, identifiers, flags, function names.

**Why**: a formula in a code fence is unreadable as mathematics — subscripts,
integrals, sums and fractions all collapse into one line of ASCII, and
`amp = std(IFFT[FFT(y-ȳ)|OPL∈[2,40]µm])/ȳ` is exactly the kind of thing this
project has already got wrong once by misreading. It also mislabels the content:
a code fence says "this is a program", when the thing inside is a definition a
reader has to check. And a defined symbol is what makes a number recomputable —
several metrics here have more than one reasonable definition.

**How to Apply**:
- Displayed for anything with a fraction, sum, integral or root; inline
  (`$\theta_i$`, `$R^{2}$`, `$\Delta n$`) for symbols in running prose.
- The sentence after the equation names each symbol *and its units*. The
  sentence after that says what the quantity is physically — what it would mean
  for it to be large, small, or zero.
- Metric tables get three columns: quantity, formula, physical meaning.
- Use `\lvert ... \rvert` rather than a bare `|` inside a markdown table cell,
  which would otherwise be read as a column separator.
- `\mathrm{...}` for multi-letter names (`\mathrm{OPL}`, `\mathrm{amp}`), not
  `\rm` — the same constraint a pandoc/DOCX build has.

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

---

## Writing Style (required for all prose I produce)

Write all text meant for humans — explanations, messages to me,
comments, documentation — in plain English:

- Use plain words and short sentences.
- Write complete sentences with subjects and verbs. One idea per
  sentence.
- Lead with the conclusion. When reporting a problem, first say what
  to do about it in one sentence; explain afterward.
- Do not use metaphors or idioms in technical statements. State the
  literal fact.
- Do not use invented or undefined jargon. Standard technical terms
  are fine; define any project-specific term where it first appears.
- Keep code comments short and to the point.
- Do not conjoin clauses. Do not use em-dashes. Do not use semicolons to join
  two independent statements.
- One sentence carries one logical step. If a sentence contains two steps,
  split the sentence into two sentences.
- Do not use pronouns that point backwards. Banned: "this", "that", "these",
  "those", "it", "ones", "the former", "the latter". Repeat the noun instead.
- Order every sentence cause first, effect second. Write "Since X, therefore Y".
  Do not write "Y because X".

### Working style

- Build the simplest version that works first. Add features only
  after I have seen the simple version run.
- Before making a major design decision, tell me the options and let
  me choose.
- Report results plainly. If something fails, show me the actual
  error message; do not guess that it worked.
