::: {.band}
## ECE 60141: Foundations of Computational Imaging

# Lab 4: MAP Reconstruction with Non-Gaussian Priors

[ECE 60141 course
page](https://engineering.purdue.edu/~bouman/ece60141/) · [Laboratory
index](https://cabouman.github.io/grad_labs/index.html)
:::

::: {.container}
Open all

Overview [What the lab is about, the data, and the ground rules.]{.note}

::: {.modulebody}
::: {.card}
### What This Lab Is About

**Due: Friday, Oct. 2.**

Your Lab 3 reconstruction had one clear defect. The prior was quadratic,
so it pulled hardest on the pixel pairs that differed the most, and the
pairs that differ the most are the edges. The restoration worked, and it
blurred every edge in the picture.

In this lab you replace the quadratic prior with a non-Gaussian one, the
QGGMRF, which does not blur edges. That fix has a disadvantage. The cost
function stops being quadratic, and the gradient descent you have been
using stops working well on it.

The way out is **majorization**. You build a quadratic function that
sits on top of the true cost and touches it at the current image. That
quadratic gives an exact step size with no tuning, and because it sits
above the true cost, each step lowers the true cost as well. The
optimization is then one steepest descent loop. Module 2 builds the
quadratic, and Module 3 runs the loop.

The background is Chapter 6 for non-Gaussian MRF priors and the QGGMRF
potential, and Chapter 8 for surrogate functions and majorization. The
parameter names follow the mbirtorch software, which reverses the roles
of $p$ and $q$ from the book.
:::

::: {.card}
### The Data

The same image and blur kernel as Lab 3, so that you can compare
directly with what you already have.

-   [**kodim23.pgm**](https://cabouman.github.io/grad_labs/labs/resources/kodim23.pgm),
    a 768 by 512 grayscale image, 384 KB.

    Source: image 23 of the Kodak Lossless True Color Image Suite,
    released for unrestricted use.

-   [**levin09_kernels.zip**](https://cabouman.github.io/grad_labs/labs/resources/levin09_kernels.zip),
    the eight camera shake blur kernels, 14 KB. This lab uses kernel 1,
    the same one as Lab 3.

    Source: A. Levin, Y. Weiss, F. Durand, and W. T. Freeman,
    [\"Understanding and Evaluating Blind Deconvolution
    Algorithms,\"](https://webee.technion.ac.il/people/anat.levin/papers/deconvLevinEtalCVPR09.pdf)
    IEEE CVPR 2009.

::: {.codeblock}
Copy

    wget https://cabouman.github.io/grad_labs/labs/resources/kodim23.pgm
    wget https://cabouman.github.io/grad_labs/labs/resources/levin09_kernels.zip
    unzip levin09_kernels.zip
:::
:::

::: {.card}
### Ground Rules

-   All code in **PyTorch**, in a single file named `lab4.py`, holding
    functions and nothing else. Run your experiments from a separate
    script.
-   You may import `forward`, `adjoint`, `load_pgm`, and `load_kernel`
    from your `lab3.py`. Those four are done and you should not write
    them again.
-   Every function has the exact signature given here, and a docstring
    saying what it does, what its inputs are, and what it returns.
-   Everything in `float64`, so that the cost differences you plot are
    not lost in rounding. No `for` loop over pixels.
-   Do not enforce a nonnegativity constraint on $x$. Leave it
    unconstrained, so that the cost you plot is the cost of equation
    (1).
-   Fix the random seed for the noise and report it.

::: {.tip}
**How you will work.** You are not going to hand-write this
implementation. You will state the math to an agentic AI, have it write
the PyTorch code, then verify it, run the experiments, and interpret the
results. Your job is to be precise about what you ask for and skeptical
about what comes back. In Step 6 you check that the true cost decreases
at every iteration, which is the test that the whole method is working.
:::
:::
:::

Parameters for the Experiments [The settings for every hand-in item, in
one place.]{.note}

::: {.modulebody}
::: {.card}
### Parameters for Every Experiment

Every reconstruction in this lab uses the same image and blur. The image
is `kodim23.pgm` on $\lbrack 0,1\rbrack$, and $A$ is convolution with
Levin kernel 1 and circular boundaries. The potential always uses
$p = 2.0$ and $q = 1.2$, and $x$ is left unconstrained.

::: {.tip}
**You generate the noise.** Draw a noise image $w$ with standard
deviation $\sigma_{w}$, add it to the blurred signal to form the data
$y = Ax_{\text{true}} + w$, and reconstruct from that $y$. Fix and
report the random seed. The symbol $\sigma_{w}$ does two separate jobs,
set to one value in this lab: it is the standard deviation of the noise
you add, and it is the noise level the cost assumes in its data term
$\frac{1}{2\sigma_{w}^{2}}\| y - Ax\|^{2}$. In general these two need
not be equal.
:::

The table lists the settings for each hand-in item. Every experiment
below points back to this table by its deliverable number, so you do not
have to search the text for a number.

  --------- --------------------------------------- -------------------- ------------ ----------------------------- ------------------------
  Item      Experiment                              Noise $\sigma_{w}$   $T$          $\sigma_{x}$                  Iterations
  D1        Potential & influence plots (Task 2a)   none \*              1            0.2 \*                        plot only
  D3        Surrogate plot (Task 4b)                none \*              1            0.2 \*                        plot only
  D5        Cost decreases each step (Task 6a)      0.02                 1            0.020                         $N = 50$, $\omega = 1$
  D6--D10   Reconstruction runs R1--R6 (Step 7)     0.02                 1 or 100 †   see the run table in Step 7   $N = 50$, $\omega = 1$
  --------- --------------------------------------- -------------------- ------------ ----------------------------- ------------------------

\* The two shape plots (D1, D3) draw $\rho$ as a function of $\Delta$
and use $\sigma_{x} = 0.2$ only to make the potential\'s two regimes
visible. They are not reconstructions and use no data. The
reconstructions use the smaller $\sigma_{x}$ values in the run table.\
† $T = 1$ is the QGGMRF (edge preserving); $T = 100$ is the Gaussian
MRF, the same potential made quadratic over the whole range.
:::
:::

Module 1: The Prior That Does Not Blur Edges [Steps 1 and 2. What goes
wrong with a quadratic, and the potential that fixes it.]{.note}

::: {.modulebody}
::: {.card}
### Step 1: Why a Gaussian Prior Blurs Edges

Write the MAP cost function in the general pairwise form

$$\begin{matrix}
{f(x) = \frac{1}{2\sigma_{w}^{2}}\| y - Ax\|^{2} + \frac{1}{2}\sum\limits_{s}\sum\limits_{r \in \partial s}b_{s,r}\,\rho\!\left( x_{s} - x_{r} \right)\ ,} \\
\end{matrix}$$

where $\partial s$ is the set of neighbors of pixel $s$, $b_{s,r}$ are
the neighbor weights, and $\rho$ is the **potential function**. The
double sum runs over each pixel and its neighbors, so every neighboring
pair appears twice, once from each end, and the $\frac{1}{2}$ corrects
for that. This is the form that maps directly onto the 8-plane neighbor
tensor you build below. In Lab 3 you used the quadratic potential

$$\begin{matrix}
{\rho(\Delta) = \frac{\Delta^{2}}{2\sigma_{x}^{2}}\ ,\qquad\rho^{\prime}(\Delta) = \frac{\Delta}{\sigma_{x}^{2}}\ .} \\
\end{matrix}$$

The derivative $\rho^{\prime}$ is called the **influence function**,
because it says how hard the prior pulls two neighboring pixels together
when their difference is $\Delta$.

Now look at what (2) does. Its influence grows without bound. The larger
the difference between two neighboring pixels, the harder the prior
pulls them together. But a real edge is exactly a place where two
neighboring pixels ought to differ a lot. The quadratic prior therefore
smooths hardest precisely where it should not, and the edges are
blurred.

What we want instead is a potential that behaves like a quadratic for
small differences, so that noise in flat regions is still suppressed,
but whose influence flattens out for large differences, so that edges
are left alone.
:::

::: {.card}
### Step 2: The QGGMRF Potential

The QGGMRF potential does exactly that. In the scaled form from Chapter
6,

$$\begin{matrix}
{\rho(\Delta) = \frac{|\Delta|^{p}}{p\,\sigma_{x}^{p}}\left( \frac{\left| \frac{\Delta}{T\sigma_{x}} \right|^{q - p}}{1 + \left| \frac{\Delta}{T\sigma_{x}} \right|^{q - p}} \right)\ ,} \\
\end{matrix}$$

with influence function

$$\begin{matrix}
{\rho^{\prime}(\Delta) = \frac{|\Delta|^{p - 1}}{\sigma_{x}^{p}}\,\frac{\left| \frac{\Delta}{T\sigma_{x}} \right|^{q - p}\left( \frac{q}{p} + \left| \frac{\Delta}{T\sigma_{x}} \right|^{q - p} \right)}{\left( 1 + \left| \frac{\Delta}{T\sigma_{x}} \right|^{q - p} \right)^{2}}\,{sign}(\Delta)\ .} \\
\end{matrix}$$

There are three parameters plus the scale $\sigma_{x}$, and each one
does a single job.

-   $\sigma_{x}$ sets the overall scale of the pixel differences the
    model expects. It controls how much regularization you get.
-   $p$ sets the shape for small differences,
    $|\Delta| \ll T\sigma_{x}$, where
    $\rho(\Delta) \approx |\Delta|^{p}/(p\,\sigma_{x}^{p})$. We use
    $p = 2$ throughout, so the potential is quadratic near zero.
-   $q$ sets the shape for large differences,
    $|\Delta| \gg T\sigma_{x}$, where
    $\rho(\Delta) \approx |\Delta|^{q}/(p\, T^{q - p}\sigma_{x}^{q})$.
    Smaller $q$ means less penalty on big jumps, so edges survive.
-   $T$ sets where the transition between the two regimes happens, at
    $|\Delta| \approx T\sigma_{x}$.

Requiring $1 \leq q < p \leq 2$ keeps $\rho$ convex. With $p = 2$ the
potential has a continuous, bounded second derivative at $\Delta = 0$,
which Module 2 needs. The plain GGMRF potential
$|\Delta|^{q}/(q\sigma_{x}^{q})$ with $q < 2$ does not have that
property, since its second derivative blows up at zero.

  ------------------- ---------------------------- ----------------------------------------------------------- -----------
  Regime              Condition                    Behavior of $\rho$                                          Exponent
  Small differences   $|\Delta| \ll T\sigma_{x}$   $\rho \approx |\Delta|^{2}/(2\sigma_{x}^{2})$, quadratic    $p = 2$
  Large differences   $|\Delta| \gg T\sigma_{x}$   $\rho \approx |\Delta|^{q}/(p\, T^{q - p}\sigma_{x}^{q})$   $q = 1.2$
  ------------------- ---------------------------- ----------------------------------------------------------- -----------

::: {.tip}
**The names of $p$ and $q$ are reversed from the book.** This lab
follows the mbirtorch software convention, in which $p$ is the exponent
near zero and $q$ is the exponent for large differences, with $p > q$.
The book (FCI) uses the opposite names, $q$ near zero and $p$ for large
differences, with $q > p$. Equation (3) is the same function in both
conventions. Only the names of the two exponents are exchanged. The
normalization stays $1/p$ either way, so the two conventions differ by
the constant factor $(q/p)\, T^{p - q}$, which is the same as a small
change in $\sigma_{x}$.
:::

::: {.tip}
**A common mistake.** Setting $p = 2$ and $q = 2$ does not give you back
the Gaussian prior of equation (2). It gives you half of it. To compare
against the Gaussian prior, use equation (2) directly.
:::

Write these two functions.

::: {.codeblock}
Copy

    def rho(delta, sigma_x, p, q, T):
        """Return the QGGMRF potential of equation (3), elementwise.

        delta    float64 tensor of any shape
        returns  float64 tensor of the same shape
        """


    def rho_prime(delta, sigma_x, p, q, T):
        """Return the QGGMRF influence function of equation (4), elementwise.

        delta    float64 tensor of any shape
        returns  float64 tensor of the same shape
        """
:::

**Task 2a.** Make two plots over $\Delta \in \lbrack - 0.3,0.3\rbrack$,
using $p = 2.0$, $q = 1.2$, $T = 1.0$, and $\sigma_{x} = 0.2$. In the
first, plot the potential $\rho(\Delta)$ for the quadratic potential (2)
and for the QGGMRF (3) on the same axes. In the second, plot the
influence function $\rho^{\prime}(\Delta)$ for the quadratic potential
and for the QGGMRF (4) on the same axes.

::: {#D1 .handin}
[HAND IN · D1]{.tag}

The two plots, the potentials and the influence functions, each showing
the quadratic and the QGGMRF, and two sentences using them to say what
each prior does to a pixel pair straddling a strong edge.
:::
:::
:::

Module 2: Majorization [Steps 3 to 4. Build the quadratic that stands in
for the true cost.]{.note}

::: {.modulebody}
::: {.card}
### Step 3: What a Surrogate Function Is

Put the QGGMRF potential into the cost (1) and two things go wrong for
plain gradient descent.

First, there is no longer one curvature. The second derivative of $\rho$
is about $1/\sigma_{x}^{2}$ near $\Delta = 0$ and falls toward zero for
large $|\Delta|$. Across an image the neighboring pixel differences span
both regimes at once, and gradient descent has one step size for the
whole image. It must be small enough for the stiffest part of the cost,
which makes it slow everywhere else.

Second, the safe step size depends on the current image, so it changes
from iteration to iteration. There is no fixed step size you can compute
once and trust.

Majorization fixes both. Instead of minimizing the hard function, we
repeatedly minimize an easy one built to stand in for it.

Let $f(x)$ be the function we want to minimize and let $x^{\prime}$ be
the current image, called the **point of approximation**. A function
$q(x;x^{\prime})$ is a **surrogate function** for $f$ at $x^{\prime}$ if

$$\begin{matrix}
{f(x^{\prime}) = q(x^{\prime};x^{\prime})\ ,\qquad f(x) \leq q(x;x^{\prime})\quad\text{for\ all~}x\ .} \\
\end{matrix}$$

In words, the surrogate touches the true cost at the current image and
lies above it everywhere else. Picture two curves that touch at one
point, with the surrogate sitting on top.

In practice we build a $Q(x;x^{\prime})$ that differs from
$q(x;x^{\prime})$ by a term depending on $x^{\prime}$ but not on $x$.
That changes nothing, because a constant does not move a minimum. Stated
in terms of $Q$, the condition is

$$\begin{matrix}
{f(x)\  \leq \ Q(x;x^{\prime}) - Q(x^{\prime};x^{\prime}) + f(x^{\prime})\quad\text{for\ all~}x,x^{\prime}\ .} \\
\end{matrix}$$

This inequality is the defining condition a surrogate must satisfy.

The **majorization-minimization** algorithm is the obvious one. From the
current image, build the surrogate, minimize it, and use the result as
the new current image:

$$\begin{matrix}
{x^{(k + 1)} = \arg\min\limits_{x}\ Q\!\left( x;x^{(k)} \right)\ .} \\
\end{matrix}$$

Here is the whole argument for why it works, in one line.

$$\begin{matrix}
{f\!\left( x^{(k + 1)} \right)\  \leq \ q\!\left( x^{(k + 1)};x^{(k)} \right)\  \leq \ q\!\left( x^{(k)};x^{(k)} \right)\  = \ f\!\left( x^{(k)} \right)\ .} \\
\end{matrix}$$

Read it left to right. The first inequality is the upper bound in (5).
The second holds because $x^{(k + 1)}$ minimizes the surrogate. The
equality is the touching condition in (5). Together they give
$f(x^{(k + 1)}) \leq f(x^{(k)})$, so the true cost never goes up.

Three consequences are worth stating plainly.

-   **You never evaluate the true cost inside the algorithm.** The
    guarantee is structural. It is not the result of a line search on
    the true cost or a backtracking test.
-   **You do not have to fully minimize the surrogate.** Look at the
    middle inequality in (8). All you need is to *decrease* the
    surrogate. Any $x$ with $Q(x;x^{\prime}) < Q(x^{\prime};x^{\prime})$
    already gives $f(x) < f(x^{\prime})$. One gradient step is enough to
    make progress on the true cost, and that is what makes the method
    cheap.
-   **A fixed point of the iteration has zero true gradient.** The
    surrogate and the true cost are tangent at $x^{\prime}$, so they
    have the same gradient there. Our cost is convex, so that fixed
    point is the global minimum. The algorithm converges to the right
    answer, not to an answer that depends on which surrogate you picked.
:::

::: {.card}
### Step 4: Building the Surrogate

The data term $\frac{1}{2\sigma_{w}^{2}}\| y - Ax\|^{2}$ is already
quadratic, so leave it alone. Only the prior term needs replacing.

Two properties from Chapter 8 make this legitimate. **Additivity** says
that if you have a surrogate for each term of a sum, the sum of the
surrogates is a surrogate for the sum. **Composition** says that if
$Q(\Delta;\Delta^{\prime})$ is a surrogate for $\rho(\Delta)$, then
substituting $\Delta = x_{s} - x_{r}$ into both gives a surrogate for
$\rho(x_{s} - x_{r})$ as a function of $x$. So the whole job reduces to
one scalar problem: find a quadratic surrogate for the scalar function
$\rho(\Delta)$. Solve it once and it applies to every neighbor pair in
the image.

We want the surrogate to be quadratic in $\Delta$, and we choose it
symmetric about zero with no linear term:

$$\begin{matrix}
{\rho(\Delta;\Delta^{\prime}) = \frac{a_{2}}{2}\,\Delta^{2}\ ,} \\
\end{matrix}$$

where $a_{2}$ depends on the current difference
$\Delta^{\prime} = x_{s}^{\prime} - x_{r}^{\prime}$. This form is chosen
on purpose: the surrogate prior term then looks exactly like a Gaussian
prior term, which is what lets you reuse Lab 3.

There is one free number and one condition that fixes it. The surrogate
must be tangent to $\rho$ at $\Delta = \Delta^{\prime}$. Matching
derivatives, $a_{2}\Delta^{\prime} = \rho^{\prime}(\Delta^{\prime})$,
gives

$$\begin{matrix}
{a_{2} = \frac{\rho^{\prime}(\Delta^{\prime})}{\Delta^{\prime}}\ ,} \\
\end{matrix}$$

and therefore the **symmetric bound surrogate**

$$\begin{matrix}
{\rho(\Delta;\Delta^{\prime}) = \left\{ \begin{matrix}
{\frac{\rho^{\prime}(\Delta^{\prime})}{2\Delta^{\prime}}\,\Delta^{2}} & {\text{if~}\Delta^{\prime} \neq 0\ ,} \\
{\frac{\rho^{\prime\prime}(0)}{2}\,\Delta^{2}} & {\text{if~}\Delta^{\prime} = 0\ .} \\
\end{matrix} \right.} \\
\end{matrix}$$

The second case is the limit of the first as
$\Delta^{\prime} \rightarrow 0$. This is where $p = 2$ is required. For
the QGGMRF, $\rho^{\prime\prime}(0)$ is finite and equals
$1/\sigma_{x}^{2}$. For the total variation potential $|\Delta|$, and
for the GGMRF with a single exponent below 2, that quantity is infinite
and the whole method fails.

Equation (11) is the surrogate written in the $Q$ form of Step 3, with
the constant dropped. Shifted back to touch the potential, it is

$$\frac{\rho^{\prime}(\Delta^{\prime})}{2\Delta^{\prime}}\left( \Delta^{2} - \Delta^{\prime 2} \right) + \rho(\Delta^{\prime})\ ,$$

which touches $\rho$ at the two symmetric points
$\Delta = \pm \Delta^{\prime}$ and lies above it everywhere else.
Touching at two points, not one, makes it a much tighter bound than the
maximum curvature surrogate, which uses the largest second derivative of
$\rho$ anywhere and is too conservative. The term it adds depends only
on $\Delta^{\prime}$, so it drops out of the cost, and only the
$\Delta^{2}$ coefficient carries into the weights below.

::: {.tip}
**The upper bound is a theorem.** It holds for any symmetric potential
whose influence function is increasing near zero and concave for
$\Delta > 0$. The QGGMRF satisfies this. Your plot in Task 4b shows it
holds rather than proving it.
:::

Now substitute (11) into the prior term of (1). Each neighbor pair
contributes a weight times $(x_{s} - x_{r})^{2}$, so collect the weights
into one coefficient per pair:

$$\begin{matrix}
{{\overset{\sim}{b}}_{s,r}\ \leftarrow\ b_{s,r}\,\frac{\rho^{\prime}(x_{s}^{\prime} - x_{r}^{\prime})}{2\,(x_{s}^{\prime} - x_{r}^{\prime})}\ .} \\
\end{matrix}$$

The surrogate MAP cost is then

$$\begin{matrix}
{Q(x;x^{\prime}) = \frac{1}{2\sigma_{w}^{2}}\| y - Ax\|^{2} + \frac{1}{2}\sum\limits_{s}\sum\limits_{r \in \partial s}{\overset{\sim}{b}}_{s,r}\,\left( x_{s} - x_{r} \right)^{2}\ .} \\
\end{matrix}$$

Compare (13) with the Lab 3 cost. They are the same function. The only
difference is that the neighbor weights are no longer fixed constants.
They are recomputed from the current image at the start of each
iteration.

::: {.tip}
**What the re-weighting does.** Majorization here is a rule for
re-weighting the neighbors of every pixel, once per iteration, so that a
quadratic prior imitates a non-quadratic one. Where two neighbors are
nearly equal, ${\overset{\sim}{b}}_{s,r}$ is at its largest and the
prior smooths hard. Across an edge, ${\overset{\sim}{b}}_{s,r}$ is small
and that pair is barely coupled. The algorithm is building a Gaussian
prior whose weights have been switched off across the edges of the
current image.
:::

For the QGGMRF, substituting (4) into (12) gives the coefficient you
will implement:

$$\begin{matrix}
{{\overset{\sim}{b}}_{s,r}\ \leftarrow\ b_{s,r}\,\frac{|\Delta^{\prime}|^{p - 2}}{2\sigma_{x}^{p}}\;\frac{\left| \frac{\Delta^{\prime}}{T\sigma_{x}} \right|^{q - p}\left( \frac{q}{p} + \left| \frac{\Delta^{\prime}}{T\sigma_{x}} \right|^{q - p} \right)}{\left( 1 + \left| \frac{\Delta^{\prime}}{T\sigma_{x}} \right|^{q - p} \right)^{2}}\ ,\qquad\Delta^{\prime} = x_{s}^{\prime} - x_{r}^{\prime}\ ,} \\
\end{matrix}$$

and, for $p = 2$, the limiting value at $\Delta^{\prime} = 0$:

$$\begin{matrix}
{{\overset{\sim}{b}}_{s,r}\ \leftarrow\ \frac{b_{s,r}}{2\,\sigma_{x}^{2}}\ .} \\
\end{matrix}$$

For the Gaussian prior of (2), the same formula (12) gives
${\overset{\sim}{b}}_{s,r} = b_{s,r}/(2\sigma_{x}^{2})$, a constant that
never changes. The Gaussian case is the special case of this algorithm
in which the surrogate is exact and the re-weighting does nothing. Your
code handles both with the same loop.

::: {.tip}
**Numerical warning.** Since $q < p$, the factor
$|\Delta^{\prime}/(T\sigma_{x})|^{q - p}$ in (14) goes to infinity as
$\Delta^{\prime} \rightarrow 0$, even though the product as a whole has
the finite limit (15). Evaluating (14) directly at small
$\Delta^{\prime}$ produces overflow or NaN. Switch to (15) whenever
$|\Delta^{\prime}|$ is below a small threshold, and make sure the switch
does not leave a NaN sitting in the unused branch. In PyTorch,
`torch.where` evaluates both branches, so clamp $|\Delta^{\prime}|$ at a
small floor, for example $10^{- 8}$ and at or below that threshold,
before forming that power rather than relying on the `where` to protect
you.
:::

**Task 4a.** Derive equation (15) from equation (14). Take the limit
$\Delta^{\prime} \rightarrow 0$ with $p = 2$, showing the step where the
two powers of $\Delta^{\prime}$ cancel. This is a paper-and-pencil
derivation, and it is the fastest way to understand why the code needs
the threshold.

::: {#D2 .handin}
[HAND IN · D2]{.tag}

Your derivation of equation (15) from equation (14).
:::

**Task 4b.** Plot $\rho(\Delta)$ from (3) over
$\Delta \in \lbrack - 0.3,0.3\rbrack$ at $p = 2.0$, $q = 1.2$,
$T = 1.0$, $\sigma_{x} = 0.2$. On the same axes plot the touching
surrogate
$\frac{\rho^{\prime}(\Delta^{\prime})}{2\Delta^{\prime}}(\Delta^{2} - \Delta^{\prime 2}) + \rho(\Delta^{\prime})$
for $\Delta^{\prime} = 0.02$ and for $\Delta^{\prime} = 0.15$, and mark
the points $\Delta = \pm \Delta^{\prime}$.

::: {#D3 .handin}
[HAND IN · D3]{.tag}

The plot of the potential with its two surrogates, and one sentence
stating whether each surrogate touches the potential at the two marked
points and lies above it everywhere else.
:::
:::
:::

Module 3: Optimization [Step 5. One steepest descent loop with an exact,
tuning free step size.]{.note}

::: {.modulebody}
::: {.card}
### Step 5: The Steepest Descent Loop

The optimization is one loop of $N$ iterations. Each iteration builds a
fresh surrogate at the current image, takes the gradient of the true
cost from `torch.autograd`, and uses the surrogate only to set the step
size. There is no inner loop.

The gradient of the true cost $f$ has a closed form. At pixel $s$,

$$\begin{matrix}
{\left\lbrack \nabla f(x) \right\rbrack_{s} = - \frac{1}{\sigma_{w}^{2}}\left\lbrack A^{t}(y - Ax) \right\rbrack_{s} + \sum\limits_{r \in \partial s}b_{s,r}\,\rho^{\prime}(x_{s} - x_{r})\ .} \\
\end{matrix}$$

You do not code this gradient. You compute it with `torch.autograd`
applied to `true_cost`, and you use equation (16) only as a reference to
check your autograd gradient against. Both terms of (16) are cheap. The
first is the forward and adjoint pair from Lab 3. The second is the
influence function $\rho^{\prime}$ summed over the 8-point neighborhood.

**The surrogate sets the step size.** Build the quadratic surrogate $Q$
at the current image, as in Step 4. Because $Q$ is quadratic, the step
that minimizes it along the gradient direction has a closed form, and it
needs no tuning:

$$\begin{matrix}
{\alpha^{\star} = \frac{g^{t}g}{g^{t}Hg}\ ,\qquad g = \nabla f(x)\ ,} \\
\end{matrix}$$

where $H$ is the Hessian of $Q$. You never form $H$. The one quantity
you need is the quadratic form in the denominator, built from pieces you
already have:

$$\begin{matrix}
{g^{t}Hg = \frac{1}{\sigma_{w}^{2}}\| Ag\|^{2} + \sum\limits_{s}\sum\limits_{r \in \partial s}{\overset{\sim}{b}}_{s,r}\,(g_{s} - g_{r})^{2}\ .} \\
\end{matrix}$$

The first term is one forward projection of the gradient. The second is
the same neighbor difference you already compute, applied to $g$ in
place of $x$, with the surrogate weights ${\overset{\sim}{b}}_{s,r}$
built at the current image. There is no factor of $\frac{1}{2}$ on it:
the Hessian of the $\frac{1}{2}$ sum in (13) carries a factor of two
that cancels it.

**Over-relaxation.** $\alpha^{\star}$ is exact for the surrogate, not
for the true cost. You may lengthen it by any factor $0 < \omega < 2$
and take the step $x\leftarrow x - \omega\,\alpha^{\star}g$. For a
quadratic you can overshoot the minimum by up to a factor of 2 and still
lower it, so the surrogate still goes down, and because the surrogate
lies above the true cost, the true cost goes down too. $\omega = 1$ is
the plain surrogate step.

::: {style="text-align:center; margin: 14px 0 6px 0;"}
![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgNDYwIDIzMiIgc3R5bGU9IndpZHRoOjEwMCU7IG1heC13aWR0aDo0MjBweDsgaGVpZ2h0OmF1dG87IiBmb250LWZhbWlseT0iSGVsdmV0aWNhLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMiIgYXJpYS1sYWJlbD0iY29zdCBhbG9uZyB0aGUgc3RlcCBkaXJlY3Rpb24gdmVyc3VzIG9tZWdhIj4KICAgICAgICA8cmVjdCB4PSI0MCIgeT0iMjAiIHdpZHRoPSIzNDUuNSIgaGVpZ2h0PSIxODAiIGZpbGw9IiMwMDAiIG9wYWNpdHk9IjAuMDQiPjwvcmVjdD4KICAgICAgICA8bGluZSB4MT0iNDAiIHkxPSIxNiIgeDI9IjQwIiB5Mj0iMjAwIiBzdHJva2U9IiM1NTUiPjwvbGluZT4KICAgICAgICA8cG9seWdvbiBwb2ludHM9IjQwLDE0IDM2LDIzIDQ0LDIzIiBmaWxsPSIjNTU1Ij48L3BvbHlnb24+CiAgICAgICAgPGxpbmUgeDE9IjQwIiB5MT0iMjAwIiB4Mj0iNDUyIiB5Mj0iMjAwIiBzdHJva2U9IiM1NTUiPjwvbGluZT4KICAgICAgICA8cG9seWdvbiBwb2ludHM9IjQ1NCwyMDAgNDQ1LDE5NiA0NDUsMjA0IiBmaWxsPSIjNTU1Ij48L3BvbHlnb24+CiAgICAgICAgPGxpbmUgeDE9IjQwIiB5MT0iNjAiIHgyPSI0MjAiIHkyPSI2MCIgc3Ryb2tlPSIjOTk5IiBzdHJva2UtZGFzaGFycmF5PSI0IDMiPjwvbGluZT4KICAgICAgICA8bGluZSB4MT0iMzg1LjUiIHkxPSI2MCIgeDI9IjM4NS41IiB5Mj0iMjAwIiBzdHJva2U9IiM5OTkiIHN0cm9rZS1kYXNoYXJyYXk9IjIgMyI+PC9saW5lPgogICAgICAgIDxwb2x5bGluZSBmaWxsPSJub25lIiBzdHJva2U9IiMyYjZjYjAiIHN0cm9rZS13aWR0aD0iMi4yIiBwb2ludHM9IjQwLDYwIDc0LjUsODEuNiAxMDkuMSw5OC40IDE0My42LDExMC40IDE3OC4yLDExNy42IDIxMi43LDEyMCAyNDcuMywxMTcuNiAyODEuOCwxMTAuNCAzMTYuNCw5OC40IDM1MC45LDgxLjYgMzg1LjUsNjAgNDIwLDMzLjYiPjwvcG9seWxpbmU+CiAgICAgICAgPHBvbHlsaW5lIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2MwMzkyYiIgc3Ryb2tlLXdpZHRoPSIyLjIiIHBvaW50cz0iNDAsNjAgNzQuNSw4NS42IDEwOS4xLDEwNi45IDE0My42LDEyMy45IDE3OC4yLDEzNi43IDIxMi43LDE0NS4yIDI0Ny4zLDE0OS41IDI4MS44LDE0OS41IDMxNi40LDE0NS4yIDM1MC45LDEzNi43IDM4NS41LDEyMy45IDQyMCwxMDYuOSI+PC9wb2x5bGluZT4KICAgICAgICA8Y2lyY2xlIGN4PSI0MCIgY3k9IjYwIiByPSIzIiBmaWxsPSIjMzMzIj48L2NpcmNsZT4KICAgICAgICA8dGV4dCB4PSI0NiIgeT0iNTQiIGZpbGw9IiM2NjYiPnN0YXJ0aW5nIGNvc3Q8L3RleHQ+CiAgICAgICAgPHRleHQgeD0iMjMyIiB5PSI0MCIgZmlsbD0iIzJiNmNiMCI+c3Vycm9nYXRlPC90ZXh0PgogICAgICAgIDx0ZXh0IHg9IjIzMiIgeT0iMTcyIiBmaWxsPSIjYzAzOTJiIj50cnVlIGNvc3Q8L3RleHQ+CiAgICAgICAgPHRleHQgeD0iNDAiIHk9IjEyIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjNTU1Ij5jb3N0PC90ZXh0PgogICAgICAgIDx0ZXh0IHg9IjQ1MCIgeT0iMjE2IiBmaWxsPSIjNTU1Ij7PiTwvdGV4dD4KICAgICAgICA8dGV4dCB4PSI0MCIgeT0iMjE1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjMzMzIj4wPC90ZXh0PgogICAgICAgIDx0ZXh0IHg9IjIxMi43IiB5PSIyMTUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiMzMzMiPjE8L3RleHQ+CiAgICAgICAgPHRleHQgeD0iMzg1LjUiIHk9IjIxNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzMzMyI+MjwvdGV4dD4KICAgICAgPC9zdmc+)
:::

Along the step direction, the surrogate is least at $\omega = 1$ and
back to its starting value at $\omega = 2$. The true cost stays below
it, so any $\omega$ in $(0,2)$ lowers the true cost.

::: {.codeblock}
Copy

    Steepest descent with a surrogate step size

      Given: y, A, sigma_w, sigma_x, p, q, T, N iterations, omega in (0, 2)
      Initialize x = y
      Record the TRUE cost f(x) from (1) and (3)
      For k = 0 to N-1:
          g          = true_gradient(x)          # torch autograd of f
          alpha_star = optimal_step_size(x, g)   # exact step for the surrogate at x
          x          = x - omega * alpha_star * g
          Record the TRUE cost f(x) from (1) and (3)
:::

The neighborhood is the 8-point one from Lab 3, with $b_{s,r} = 1/6$ for
the four horizontal and vertical neighbors and $b_{s,r} = 1/12$ for the
four diagonal neighbors, so that
$\sum\limits_{r \in \partial s}b_{s,r} = 1$ at an interior pixel. As in
Lab 3, a pair counts only when both pixels lie inside the image.

To keep everybody\'s code comparable, the neighbor differences and the
surrogate weights are stored as $(8,H,W)$ tensors, one plane per
neighbor offset, in this fixed order.

::: {.codeblock}
Copy

    NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]

    B_WEIGHTS = [1/12, 1/6, 1/12,
                 1/6,        1/6,
                 1/12, 1/6, 1/12]


    def neighbor_diffs(x):
        """Return the differences x[s] - x[r] for the 8 neighbor offsets.

        Plane i holds x minus x shifted by NEIGHBORS[i].  Entries whose
        neighbor falls outside the image are set to zero, and the matching
        entries of the mask are zero, so that a pair counts only when both
        pixels are inside.

        x        (H, W) float64 tensor
        returns  (diffs, valid), each (8, H, W), float64 and bool
        """


    def surrogate_weights(x, sigma_x, p, q, T):
        """Return the b_tilde coefficients of equations (14) and (15).

        Used inside optimal_step_size to form the surrogate Hessian.
        returns  (8, H, W) float64 tensor, zero where the pair is invalid
        """


    def true_cost(x, y, a, sigma_w, sigma_x, p, q, T):
        """Return the true cost f(x) of equation (1).

        Differentiable in x, so torch.autograd can take its gradient.  Floor
        |x_s - x_r| at a small value (for example 1e-8) before the QGGMRF
        power: rho is NaN at Delta = 0 in the forward value too, not only in
        autograd, because |Delta/(T sigma_x)|^(q-p) is infinite there.  Apply
        the neighbor validity mask after rho so floored border pairs contribute
        exactly zero.

        returns  float64 scalar tensor
        """


    def true_gradient(x, y, a, sigma_w, sigma_x, p, q, T):
        """Return the gradient of the true cost, equation (16).

        Computed with torch.autograd.grad(true_cost(x, ...), x).  x must have
        requires_grad set, and the forward and adjoint of a must be torch
        operations so the graph reaches x.  Equation (16) is the analytic form
        to check it against.

        returns  (H, W) float64 tensor
        """


    def optimal_step_size(x, g, a, sigma_w, sigma_x, p, q, T):
        """Return the exact surrogate step size alpha* of equations (17)-(18).

        Builds the surrogate weights at x and uses one projection A g to form
        the denominator g^t H g.  g is the gradient from true_gradient.  The
        denominator uses only A g, the surrogate weights at x, and sigma_w; it
        does not use y.

        returns  float64 scalar tensor
        """


    def reconstruct(y, a, sigma_w, sigma_x, p, q, T, num_iters, omega):
        """Steepest descent with the surrogate step size, starting at y.

        Each iteration: g = true_gradient(x); alpha = optimal_step_size(x, g);
        x = x - omega * alpha * g.

        returns  (x, cost_history), the final image and a list of the true
                 cost after each iteration, of length num_iters + 1 whose
                 first entry is f(y)
        """
:::
:::
:::

Module 4: Build It, Check It, Run It [Steps 6 to 8. One check that the
cost decreases, six runs, and seven questions.]{.note}

::: {.modulebody}
::: {.card}
### Step 6: Build the Code and Check It

Give your AI assistant a specification, not a description. At a minimum
it needs equations (1), (3), (4), (14), (15), (16), (17), and (18), the
neighborhood and its weights, the signatures above, and the pseudocode
in Step 5. Require float64 and no pixel loops. There is one potential,
the QGGMRF; the Gaussian MRF is the same potential at a large $T$, so
every run goes through the identical steepest descent loop and cost
reporting. If you build the code in parts, write a separate
specification for each part.

::: {#D4 .handin}
[HAND IN · D4]{.tag}

The specification, or one specification per part if you built it in
parts, that you gave your AI assistant, quoted, with a short note on
what each got wrong the first time and how you found out.
:::

**Task 6a. Check that the cost decreases.** The majorization argument
says the true cost must go down at every iteration. That is the one
thing to verify before you trust a reconstruction. Run the baseline
QGGMRF, record the true cost $f(x)$ after each iteration, and plot it
against the iteration number. If it ever rises, your surrogate weight
formula or your step size is wrong.

::: {#D5 .handin}
[HAND IN · D5]{.tag}

The plot of the true cost against the iteration number for the baseline
QGGMRF run, and one sentence stating whether it decreased at every
iteration.
:::
:::

::: {.card}
### Step 7: Experiments

Everybody runs the same problem, so the results can be compared.

**This step shows two things. First, at its best $\sigma_{x}$ the QGGMRF
($T = 1$) gives a sharper edge and a lower RMSE than the Gaussian MRF
($T = 100$). Second, each prior has a best $\sigma_{x}$: halving it or
doubling it makes the reconstruction worse.** If the QGGMRF and the
Gaussian reconstructions look the same, the QGGMRF is not working and
something is wrong.

The image is scaled to $\lbrack 0,1\rbrack$. Use `kodim23.pgm` in
$\lbrack 0,1\rbrack$ as $x_{\text{true}}$, and form

$$\begin{matrix}
{y = A\, x_{\text{true}} + w\ ,\qquad w \sim N(0,\sigma_{w}^{2}I)\ .} \\
\end{matrix}$$

Here $A$ is convolution with Levin kernel 1 and circular boundaries,
exactly as in Lab 3. Draw a noise image $w$ with standard deviation
$\sigma_{w} = 0.02$, that is, two percent of the $\lbrack 0,1\rbrack$
range, add it to $Ax_{\text{true}}$ to form $y$, and do not clip $y$.
Use the same $\sigma_{w}$ in the cost, so the reconstruction is given
the true noise level. This is the same noise level as the deblurring
problem in Lab 3, so the three labs pose the same problem. The noise is
small, so the reconstruction is limited by the blur and not by the
noise, which is where the edge-preserving prior helps most. (In the
mbirtorch software this noise standard deviation is called `sigma_y`.)

Both priors use the one QGGMRF potential. The prior is set by $T$:
$T = 1.0$ is the QGGMRF, the edge-preserving prior; $T = 100$ is the
Gaussian MRF, because at this large $T$ the potential is quadratic over
the whole range of differences in the image. Fix $p = 2.0$, $q = 1.2$,
the over-relaxation factor $\omega = 1$, and $N = 50$ iterations in
every run.

For each prior you run three values of $\sigma_{x}$: a center value,
half of it, and twice it. The center values, $0.020$ for the QGGMRF and
$0.036$ for the Gaussian MRF, are near the $\sigma_{x}$ that minimizes
RMSE for each prior.

  ----- -------------- ----- ----------------------
  Run   Prior          $T$   $\sigma_{x}$
  R1    Gaussian MRF   100   $0.036$ (center)
  R2    QGGMRF         1     $0.020$ (center)
  R3    QGGMRF         1     $0.010$ (center / 2)
  R4    QGGMRF         1     $0.040$ (center × 2)
  R5    Gaussian MRF   100   $0.018$ (center / 2)
  R6    Gaussian MRF   100   $0.072$ (center × 2)
  ----- -------------- ----- ----------------------

R1 and R2 are the best of each prior. R3 and R4 change $\sigma_{x}$
around the QGGMRF center; R5 and R6 change it around the Gaussian
center.

Pick one $128 \times 128$ crop containing a sharp, high contrast edge.
Use the same crop in every image figure and state its coordinates.

::: {#D6 .handin}
[HAND IN · D6]{.tag}

Four images with the same gray scale: $x_{\text{true}}$, $y$, the R1
Gaussian MRF reconstruction, and the R2 QGGMRF reconstruction. Then the
same four as your $128 \times 128$ crop, with the crop coordinates
stated.
:::

::: {#D7 .handin}
[HAND IN · D7]{.tag}

An edge profile plot. Pick one horizontal line crossing a strong edge in
your crop, and plot the pixel values along it for $x_{\text{true}}$, R1,
and R2 on the same axes.
:::

::: {#D8 .handin}
[HAND IN · D8]{.tag}

The $\sigma_{x}$ study. For each prior, the three crops at $\sigma_{x}$
center / 2, center, and center × 2 --- R3, R2, R4 for the QGGMRF and R5,
R1, R6 for the Gaussian MRF --- each labeled with its $\sigma_{x}$ and
its RMSE.
:::

::: {#D9 .handin}
[HAND IN · D9]{.tag}

An image of $\sum\limits_{r \in \partial s}{\overset{\sim}{b}}_{s,r}$ at
the final iteration of R2, scaled for visibility, with the display range
stated.
:::

::: {#D10 .handin}
[HAND IN · D10]{.tag}

A table with one row per run, R1 through R6, giving the final true cost,
the RMSE against $x_{\text{true}}$ in $\lbrack 0,1\rbrack$ units, and
the wall clock time.
:::
:::

::: {.card}
### Step 8: Questions to Answer

Answer each in a few sentences, pointing at your own figures and
numbers. These are the point of the lab.

1.  Compare the R1 Gaussian MRF and R2 QGGMRF crops and the edge
    profile. Describe the difference at the edge and the difference in
    the flat regions. Use your influence function plot from D1 to
    explain why the two priors treat the edge differently.
2.  Did the lower RMSE go with the reconstruction that looks better to
    you? If not, say what RMSE is failing to measure.
3.  What does $\sigma_{x}$ control? Use R3 and R4 for the QGGMRF and R5
    and R6 for the Gaussian MRF, and say what happens in the limits of
    very small and very large $\sigma_{x}$.
4.  The QGGMRF reaches its best RMSE at a smaller $\sigma_{x}$ (R2,
    $0.020$) than the Gaussian MRF does (R1, $0.036$). A smaller
    $\sigma_{x}$ is a stronger prior. Explain why the edge-preserving
    prior can regularize the flat regions this much harder without
    blurring the edge, using your influence function plot from D1.
5.  Why is $p = 2$ required by the algorithm, while $q$ is a free
    modeling choice? Answer in terms of the surrogate needing a bounded
    second derivative at $\Delta = 0$.
6.  Was the true cost monotone decreasing in R2? Say which step of the
    argument in (8) guarantees this, and what would happen to the
    guarantee if your $\overset{\sim}{b}$ formula came out slightly too
    small.
7.  Look at your image of $\sum\limits_{r}{\overset{\sim}{b}}_{s,r}$
    from D9. What image structure do you see in it, and why is it there?

::: {#D11 .handin}
[HAND IN · D11]{.tag}

Your answers to the seven questions above.
:::
:::
:::

Deliverables [What to submit, and where each item comes from.]{.note}

::: {.modulebody}
::: {.card}
### What to Hand In

Submit two files through Brightspace.

1.  A report, as a single **PDF**, labeled \"Lab 4\", containing the
    items below in this order.
2.  Your `lab4.py`, as a plain `.py` file. Do not paste the code into
    the report and do not send a zip file.

Also commit your code to your `image-processing-labs` repository under
`lab4` and push it before you submit.

1.  []{.dnum} Your name, the link to your GitHub repository, and the
    random seed you used.
2.  [[D1](#D1)]{.dnum} The potential and influence-function plots, with
    two sentences.
3.  [[D2](#D2)]{.dnum} Your derivation of equation (15) from equation
    (14).
4.  [[D3](#D3)]{.dnum} The potential with its two surrogates, and one
    sentence.
5.  [[D4](#D4)]{.dnum} The specification you gave your AI assistant, and
    what it got wrong.
6.  [[D5](#D5)]{.dnum} The cost-versus-iteration plot, with the
    monotonicity statement.
7.  [[D6](#D6)]{.dnum} The four full images and the four crops.
8.  [[D7](#D7)]{.dnum} The edge profile plot.
9.  [[D8](#D8)]{.dnum} The $\sigma_{x}$ study: three crops per prior,
    with RMSE.
10. [[D9](#D9)]{.dnum} The image of the summed surrogate weights.
11. [[D10](#D10)]{.dnum} The table of cost, RMSE, and time for R1
    through R6.
12. [[D11](#D11)]{.dnum} Your answers to the seven questions.

Keep it concise and clear.
:::

::: {.card}
You now have the two pieces that make modern model-based reconstruction
work: a prior that does not destroy edges, and a way to optimize a
non-quadratic cost using nothing but the quadratic solver you already
had. In Lab 5 the prior is replaced by a denoiser, and the same idea of
solving a hard problem through a sequence of easy ones shows up again.

[Back to the laboratory
index](https://cabouman.github.io/grad_labs/index.html) · [ECE 60141
course page](https://engineering.purdue.edu/~bouman/ece60141/)
:::
:::
:::
