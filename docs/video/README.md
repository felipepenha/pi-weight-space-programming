# Pi Weight-Space Programming (π · WSP) Terminal Demo

Automated terminal demo showcasing natural language interaction with [Pi](https://pi.dev) equipped with the Weight-Space Programming harness extension.

---

## Demo Outputs

| Format | File | Resolution | Duration | File Size | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MP4 Video** | [`demo.mp4`](demo.mp4) | 1320 × 1040 | ~48s | **3.1 MB** | High-definition H.264 video with smooth 25 fps playback |
| **Animated GIF** | [`demo.gif`](demo.gif) | 1320 × 1040 | ~48s | **4.3 MB** | Universal animated preview for embedding in GitHub docs |

---

## What the Demo Shows

1. **Pi Interactive Startup with WSP Extension**:
   - Monochromatic red gradient ANSI shadow logo matching `assets/banner.png`.
   - Diamond badge and status panel showing active WSP configuration: `Mode: tool-free (Exp 1)` matching the paper's benchmark constraint.
   - Zero-gradient guarantee: $\nabla_\theta L = 0$ (No Backprop).

2. **Real Live Run: Experiment 1 Benchmark Question**:
   - Natural language prompt from the paper:
     > *"Which game was this neural network trained on? Here are the observations from the weights: Channel 0 has large weights strictly at row 8; Channels 1 and 4 have velocity pairing across rows 1-8; Channel 2 has reward weights at rows 1-3; and only left and right actions are active."*
   - Real-time mechanistic deduction streaming directly from the model inside the Pi TUI.
   - Grounded deduction mapping weights directly to Atari mechanics:
     - **Channel 0 at row 8**: Paddle constrained to horizontal motion at the bottom row.
     - **Channel 2 at rows 1–3**: Brick arrangement yielding reward points at the top of the playfield.
     - **Channels 1 & 4 velocity pairing across rows 1–8**: Ball position and velocity differencing via stacked frames.
     - **Actions Left & Right**: Direct confirmation of **Breakout (Atari 2600)**.

---

## Reproducing or Recording the Demo

The demo is automated end-to-end using [Charm VHS](https://github.com/charmbracelet/vhs) driving a live interactive session of `pi`.

### Prerequisites

```bash
brew install vhs ffmpeg ttyd
npm install -g @mariozechner/pi-coding-agent
```

### Recording

Run the recorder script:

```bash
./docs/video/record.sh
```

Or invoke VHS directly:

```bash
vhs docs/video/demo.tape
```
