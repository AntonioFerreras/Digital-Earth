# Testing

Two layers: unit tests, which check that the code does what it says, and the cloud
bench's validation suite, which measures how close the renderer's clouds come to the
truth and how fast they get there.

## Unit tests

```sh
pixi run zig build test
```

## Cloud bench validation

The cloud bench (`src/render/CloudBench.zig`) renders one cloud volume two ways: a
path tracer, exact but slow, and the approximation, which marches the sun's single
scattering and reads high order scattering from a baked probe field. The suite scores
both halves of the approximation, the bake and the rendering, against a fixed ground
truth, and records how long each took.

### Ground truth

`validation/gt/`, kept out of git, holds:

- **Path traced images:** per view, converged, at 2048 x 2048.
- **The converged field:** the probe field baked to many walks, or photon rounds, with
  the default config.
- **The converged field, rendered:** per view, with the default config, converged.
- **The render noise floor:** the same, at the test's samples per pixel.
- `manifest.json`: when and from which commit it was made, the suite's parameters,
  the times, and the comparisons below that involve ground truth only.

The parameters (sample counts, walks, image size) are in `tests/validation/suite.json`.

**The ground truth is the desired current look, and changes only with approval.** It is
regenerated only when the owner explicitly approves it. That is warranted when the path
tracer or the probe field fundamentally improves in quality: a more faithful phase,
medium or light transport, a better representation of the field, a new default whose
look is wanted. It is not warranted when either only gets more efficient: a faster bake
or render of the same result is measured against the same truth. Agents may suggest a
regeneration and say why, and wait for the answer. Changing the views, the cloud volume
or the suite's parameters also changes the truth, and needs the same approval.

### Views

Twenty-four views of the cloud, in `tests/validation/views/`, written by
`python/validation/views.py`: six camera directions, spread evenly over the sphere
around the cloud's centre and kept above -10 degrees of elevation so that they stay
above the ground, each under four suns.

- **Framing:** four directions frame the whole cloud: each turns to the middle of the
  cloud's occupied cells as it sees them, and its field of view takes them all in with
  2 % to spare. Two are close-ups with a 10 degree field of view on the cloud's centre.
- **Suns:** four, at one azimuth, with the cosine of their zenith angle evenly spaced
  from -0.07, just set, to 1, overhead.
- **Exposure:** each view's auto exposure (`Post.autoExposure`), the viewer's too: a
  smooth curve of stops by the sun's cosine zenith at the camera, which holds the sky's
  displayed brightness from overhead down to sunset and dims it below the horizon,
  measured by `python/validation/auto_exposure.py`.

Every other setting stays at its default, so that the default config is what the views
render.

### What is compared

Images, per view:

- **A, path traced against the converged field rendered:** the error of the
  approximation itself, with the bake's noise gone. Changes only with the ground truth.
- **B, the converged field rendered against the test's field, rendered with the test's
  settings:** what the test config shows, bake and rendering together.
- **C, the converged field rendered against the test's field, rendered with the default
  config, except the settings that make the field, such as the bake and the layout:**
  the bake alone. Read it against the render noise floor, the converged field
  rendered at the test's samples, which is what C would score with a perfect bake.

Fields, against the converged field, when the test's layout and medium are the ground
truth's:

- **Sun fluence:** each probe's fitted fluence from 32 fixed sun directions, its median
  and 90th percentile relative error, and its mean ratio over deciles of the probes'
  brightness, darkest first, which shows bias.
- **Sky fluence:** under a fixed sky, brighter overhead, the same errors.
- **Flux:** the error of the probes' flux under light from every direction, relative to
  their fluence.

Convergence, of the test's field alone: the bake also writes the field at half its walks,
and the two halves' disagreement gives the relative standard deviation of the full
field's sun and sky fluence, robustly, as 1.4826 times the median absolute deviation.

### Metrics

Images are scored as displayed, with the view's exposure and the default config's tone
mapping and post-processing, from the 8-bit PNG:

- **PSNR** in dB, over every pixel and channel.
- **SSIM** (Wang et al. 2004): an 11-tap Gaussian window of 1.5 pixels, per channel,
  averaged.

**Times:** every bake and render records its milliseconds of GPU work, not the process's
setup. A run's bake time includes laying the probes out.

### Running

From the repository root, after `pixi run zig build -Doptimize=ReleaseFast`:

```sh
# Ground truth: only with the owner's approval (see above).
pixi run -e py python python/validation/validate.py gt
# Score the cloud bench settings of a viewer config.
pixi run -e py python python/validation/validate.py run --config my_settings.zon --label photons
```

A run takes the `.cloud_bench` block of the config and bakes the field it describes:
`--walks` overrides the walks, else the config's are used, else the suite's. The scene,
extinction, bounce limit and sky, stays the ground truth's, as only that compares.
Results go to `validation/runs/<time>-<label>/results.json`, with the images and the
fields; a summary line per run is appended to `validation/history.jsonl`.

The suite drives two commands of `earth`, which validation needs and which leave the
viewer's probe cache alone:

- `earth bake-probes --config <file> --out <field.eprb> [--half <field.eprb>] [--probes <field.eprb>]`
  bakes a field to the config's walks and prints the bake's time.
- `earth render --probes <field.eprb> --jobs <jobs.zon>` renders several images with a
  given field on one renderer, which spares a process's setup per image.
