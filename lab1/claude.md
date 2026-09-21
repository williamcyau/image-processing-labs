# Thingy Prompt: Rectangular Gratings Diffraction Orders Simulator

## What I Am Building

A .html web page that simulates the intensity of diffracted light versus angle of diffraction of rectangular gratings. The user is allowed to tune knobs of parameters and observe how the diffraction profile changes.

## Guidelines

These are standing rules for this project. They apply to everything
you produce.

### Writing style

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

### Working style

- Build the simplest version that works first. Add features only
  after I have seen the simple version run.
- Before making a major design decision, tell me the options and let
  me choose.
- Report results plainly. If something fails, show me the actual
  error message; do not guess that it worked.

## Features

Must have:

1. A dynamic cross-section view of a standard rectangular grating, responding to user's inputs on setup parameters.
2. A panel of sliders that allow the user to input parameters, including but not limited to:
    1. grating period, 
    2. duty cycle,
    3. groove thickness,
    4. substrate thickness,
    5. sidewall angle of grooves,
    6. sidewall roughness,
    7. source wavelength, and
    8. incidence angle.
3. A plot of intensity versus angle of diffraction of the setup. It must satisfy:
    1. the peaks in the intensity must correlate with what the grating formula predicts,
    2. the data between the diffraction orders can be synthetic,
    3. make sure the curves are smooth.
    4. parameters other than a. grating period, b. source wavelength, and c. incidence angle should barely change the curve.
    5. plots must be labelled clearly

Nice to have:

1. ability to overlay a result on top of a previous result

## Look and Feel

The interactive demo should be neatly organized into three panels:
1. leftmost: knobs controlling setup parameters,
2. center: cross-section of grating,
3. rightmost: intensity vs. angle of diffraction curve.

## Platform

The plot should be visualized via Matplotlib. The grating orders should be calculated via NumPy.

## What Done Looks Like

I can open the web page, and tweak the parameters. As I am tweaking the parameters, I should see the grating cross-section change, and the plot change.

## Reflection

### A) Was this fun? Why or why not?

This is fun. This allows me to experience the power of agentic AI tools in conveying scientific computing and results. As a researcher, my time is dominated by problem-solving, instead of scientific communication. However, it is integral that I convey my research outcomes effectively and interactively. Previously, I did not have the resources to make a demo like this. Now with agentic AI tools, I am able to do so within a reasonable timeframe.

### B) What did I learn?

I learned that clear and concise instructions are integral to working with agentic AI tools. Multiple iterations are frequently required to move the product above the specs. 

### C) What would I do differently?

I would have begun using and experimenting with agentic AI tools earlier and tested different ways of optimizing with it.
