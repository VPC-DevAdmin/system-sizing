# Retail clothing try-on on AMD Instinct MI350P: project brief

Written 2026-11-17 as a handoff to a new session and a new, separate project.
It is self-contained: the reader is assumed to know nothing of the earlier work.

## 1. What this is

A technical marketing project for AMD: a conference booth demo. A visitor
stands in front of a camera, picks clothing, and within a second or two sees
photo-realistic images of themselves wearing it. Everything runs on one AMD
Instinct MI350P card in one server, with nothing sent to a cloud service.

The claims to prove:

- **Photo-realistic try-on in about a second or two**, on hardware a retailer
  can own.
- **A capacity figure:** try-ons per second per card, and how many kiosks or
  online shoppers one card serves.
- **A cost figure:** cost per try-on on owned hardware against the listed
  price of a hosted try-on service.
- **Shopper photos never leave the retailer's server.**

Photo-realism is the requirement that decides everything else. A result that
looks like a cartoon overlaid on a body is a failure, however fast it is.

The project stands alone. It must not depend on the capacity simulator
(`capsim`, in the `system-sizing` repository). It may borrow measured numbers
and launch settings from that work (section 7).

## 2. Audience and rules

- Output feeds technical marketing (booth demo, short write-up, sizing
  guidance), not a research paper. Harden for credibility; no research-grade
  reproducibility.
- Keep the method simple and explainable. Do not spend long test time chasing
  a metric.
- Maximise the benefit shown, never manufacture it. Show real results,
  including where the model does badly.
- **No published AMD-versus-NVIDIA comparison** (a Dell constraint). Compare
  against cloud pricing or against the system itself.
- **Off limits, already done for Intel:** video translation, AI video
  assistants, agentic CPU-to-GPU ratio, host-memory KV cache tiers, a
  real-time voice assistant. Clothing try-on is clear of all of these.
- The goal is to help AMD win deals against cloud services and NVIDIA.
  Favour things that work out of the box.
- **Scope is clothing only.** Hairstyles were considered and set aside.
- **Downloads need the user's explicit approval**, each time. No model or
  dataset named here has been approved or downloaded yet.
- sudo needs the user's password; the user runs any root step. Do not change
  the power cap, card partitioning, kernel arguments, drivers or firmware.

## 3. Hardware

Dell R7725, 2 x EPYC 9455 (96 cores, 192 threads), 377 GB RAM, **one** AMD
Instinct MI350P.

- MI350P: 144 GB HBM3E, PCIe card, 450 W cap in this server (600 W card).
- Ubuntu 24.04, ROCm 10.0. Access: `ssh amdgpu` (jumps through `xe7740`).
- Storage: `/data` (model cache at `/data/capsim/huggingface/hub`) and
  `/data2` (large files in `/data2/capsim/hf-overflow`, symlinked into the
  hub). Check free space before a download.
- The box is shared with a benchmark queue. Check that nothing is running
  before using the GPU (`docker ps`, and `curl localhost:8321/api/status`).
- A possible 8 x MI350P system later. Not confirmed, and not needed here.

Card facts that matter for this project:

- **Media decoders on the card.** 20 JPEG engines and 2 video engines, both
  working on this chip (section 7). A camera feed can be decoded straight
  into GPU memory with no copy through the host.
- **Several engine processes can share the card,** each with a memory share.
  This is how the best text throughput was reached, and it is the planned
  route to try-ons per second.
- **Memory is not the constraint.** Try-on models are small next to 144 GB.
  The memory lets many copies run at once; it is not what makes one copy
  work. Do not claim otherwise.

## 4. The booth experience

1. The visitor stands on a floor mark in front of a screen showing the live
   camera.
2. They tap a garment, or the system offers four.
3. Within a second or two the screen shows photo-realistic images of them in
   each garment, side by side.
4. A side panel shows live figures: seconds per try-on, try-ons per second
   on the card, cost against a hosted service.

Generating four outfits at once is deliberate: it makes the card's
throughput visible to the visitor.

This is a still-image demo with a live camera preview. It is not live video
try-on (section 9).

## 5. Design

| Stage | What it does | Where it runs |
|---|---|---|
| Capture | Camera frames into GPU memory | Card's decoder (rocJPEG or rocDecode) |
| Pick the frame | Wait until the person is still and facing forward; take one frame | Small pose or person-segmentation model |
| Try-on | Combine the person photo with each garment photo | Try-on model, several copies on the card |
| Display | Web page on the booth screen, with the metrics panel | Server CPU |

- Garment photos are prepared once in advance (cropped, background removed
  if the model needs it). Nothing is fetched from the internet during the
  show.
- Keep nothing after a session. Person photos live in memory only.
- A recorded fallback must exist for a camera or model failure.

### Candidate models

| Model | What it is | Licence | Status |
|---|---|---|---|
| FASHN VTON v1.5 | Purpose-built clothing try-on, released January 2026 | Apache 2.0 according to the vendor's own guide | Confirm on the model card; not downloaded |
| Qwen-Image-Edit | General image editor taking a person photo plus a reference image | Believed Apache 2.0 | Confirm on the model card; not downloaded |
| CatVTON | Rated best of the open models in one comparison | Non-commercial | Excluded |
| IDM-VTON | Well-known try-on model | Non-commercial | Excluded |

Check for newer releases before starting; this field moves monthly. A model
is usable only if its licence allows commercial use, because a marketing
demo is commercial use.

## 6. Clothing images

Product photos copied from shopping sites are copyrighted and carry
trademarks; a Dell or AMD legal reviewer would likely reject them. Use these
sources instead:

| Source | Why it works | Limit |
|---|---|---|
| Dell and AMD branded clothing | They own the images and logos; suits the booth | Small range; someone at Dell or AMD must supply photos |
| Unsplash and Pexels | Licences allow commercial use | Few flat product shots; avoid visible third-party logos |
| Wikimedia Commons, museum open-access collections | CC0 or CC BY; historical costume is a fun option | Variable quality |
| A retail partner | Real catalogue with permission; strongest story | Needs an agreement |

- 20 to 30 garments are enough. Start with tops and jackets, which try-on
  models handle best. Exclude swimwear and underwear.
- Keep a manifest: for every image, the source URL, the licence, the author
  and the date fetched. No image goes in the demo without an entry.
- The standard research try-on sets (VITON-HD, DressCode) are not licensed
  for commercial use. Do not use them, including for published quality
  scores.

## 7. What is already known

From the `system-sizing` work on this box, November 2026. Details are in
`docs/r7725-mi350p-roofline-plan.md`, section "Results to date".

- **No image-generation model has been run on this stack.** Text models run
  through vLLM and SGLang container images built for ROCm. Whether a
  diffusion model runs here, and how fast, is the first unknown.
- **JPEG decode on the card (rocJPEG), real film frames:** about 16,200
  images/s at 720p and 7,330 at 1080p at peak; 774 and 359 on one stream.
  About 35 to 40 CPU cores match the card, so the claim is "frees CPU cores
  and lands frames in GPU memory", not "faster than a CPU".
- **Video decode on the card (rocDecode):** about 3,100 frames/s of 1080p
  H.264 in total across the two engines.
- **Zero-copy into a PyTorch GPU tensor works.** Decode alongside a running
  text model showed no measurable slowdown.
- Probe code and outputs: `/data/capsim/mediaprobe` on the box.
- **Several engines on one card must start one after the other** with
  explicit memory shares. Started together, the second ran out of memory.
- **On fast text models the serving host, not the card, was the limit**
  until the serving settings were tuned. Expect the same kind of host-side
  bottleneck (image encode, web serving) once the model is fast.
- Model weights load fast (44 GB in about 8 s), but engines took about 100 s
  to become ready. Keep the models loaded for the whole show.

## 8. Phases

1. **Does it run, and how fast (half a day).** Load each candidate on the
   card. Record seconds per image, memory used and card power, and look at
   the output. **Gate:** if no commercially licensed model produces
   photo-realistic results at a few seconds per image or better, stop and
   report.
2. **Quality pass (one day).** About 10 garments on about 10 varied person
   photos under booth-like lighting. Choose the model, the garment types
   that hold up, and the resolution. Person photos must be ones we may use:
   team volunteers or commercially licensed stock.
3. **Throughput (one day).** Several copies of the model on the card.
   Measure try-ons per second, the response time as load rises, and how
   many kiosks one card supports. Try the usual speed levers (fewer steps,
   a distilled variant, half precision, compilation) only where quality
   survives a side-by-side look.
4. **Booth application (two to three days).** Capture, frame picking, the
   four-up display, the metrics panel.
5. **Hardening (one day).** A full-day soak run, the recorded fallback,
   lighting and backdrop guidance for the booth.
6. **Stretch: live video preview.** Only if a real-time video try-on model
   with a commercial licence exists and runs here (section 9).

## 9. Live video: why it is a stretch goal

Checked 2026-11-17:

- MagicTryOn: video try-on built on an open video model, four-step variant
  claimed real-time, weights released. **Non-commercial licence.**
- LiveVVT: August 2026 paper on streaming try-on. No code or licence found.
- JoyAI-Video-Edit (JD): Apache-licensed, general live video editing; its
  own notes claim 840 x 480 at 24 frames per second on one consumer NVIDIA
  GPU, with reference-image character replacement. Not try-on specific, so
  garment detail is likely weaker. Untested on AMD.

Live results today are about 480p with visible flicker and drift, which
fails the photo-realism requirement. Recheck before phase 6.

## 10. Figures to publish

| Figure | Note |
|---|---|
| Seconds from tap to image | State the resolution and the number of garments generated |
| Try-ons per second per card | State the number of model copies and the card power |
| Kiosks or shoppers served per card | State the assumed try-ons per shopper |
| Cost per try-on, owned versus hosted | Use a hosted service's public list price, with the date; state server cost and lifetime assumptions |
| CPU cores left free by on-card decode | Supporting point only |

Show a few imperfect results alongside the good ones. Do not publish a
quality score from a non-commercial test set.

## 11. Risks

- **The stack.** Image generation on AMD's software is less proven than
  text. Phase 1 may fail or be slow.
- **Photo-realism under booth conditions.** Crowds behind the visitor, bags,
  lanyards and mixed lighting all hurt. A plain backdrop, a floor mark and
  fixed lighting remove most of it.
- **Odd outputs.** Try-on models sometimes distort bodies. Limiting garment
  types reduces this; a staff member should be able to clear the screen.
- **Licences.** Model and image licences must be confirmed from the source,
  not from a third-party listing.
- **Privacy.** Taking part implies consent, but some regions regulate
  stored face images. Show a sign, keep nothing, and say so on screen.
- **Shared box.** Runs here can collide with the benchmark queue.

## 12. Open questions for the user

- Approve the phase 1 model downloads (FASHN VTON v1.5, Qwen-Image-Edit)?
- Which garment source first, and can Dell or AMD supply branded clothing
  photos?
- Will the server be on the booth floor, or remote with the booth as a thin
  client? Remote adds network delay and depends on conference internet.
- Which conference and date? That sets how many phases fit.
- Which hosted try-on service is the cost comparison?

## 13. Not in scope

Hairstyles, live video try-on (unless section 9 changes), head-to-head with
NVIDIA, training or fine-tuning a try-on model, storing visitor images.
