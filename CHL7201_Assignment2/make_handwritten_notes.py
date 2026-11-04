#!/usr/bin/env python3
"""Plain-paper notes for CHL7201 Assignment 2, questions 1 to 4.

Matched to the Assignment 1 scan: blue ballpoint, rightward slant, plain
white paper, and a centered underlined heading on the first page only.
Equations use the same pen as the words, so the numbers stay exact.
"""

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT_REG = ROOT / "fonts" / "Caveat.ttf"
OUT = ROOT / "Assignment2_CHL7201_handwritten.pdf"

font_manager.fontManager.addfont(str(FONT_REG))
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Caveat"
matplotlib.rcParams["mathtext.it"] = "Caveat"
matplotlib.rcParams["mathtext.bf"] = "Caveat"
matplotlib.rcParams["mathtext.cal"] = "Caveat"
matplotlib.rcParams["mathtext.tt"] = "Caveat"
matplotlib.rcParams["mathtext.sf"] = "Caveat"

# A4 at 160 dpi. Plain white sheet, blue ballpoint.
W, H = 1323, 1870
DPI = 160
INK = (36, 54, 130)
INK_HEX = "#243682"
PAPER = (255, 255, 255)
MARGIN_L = 86
MARGIN_R = 70
CONTENT_W = W - MARGIN_L - MARGIN_R
LINE0 = 48
LINE_H = 54
BOTTOM = H - 64

_FONT = {}
_MATH = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONT:
        _FONT[key] = ImageFont.truetype(str(FONT_REG), size)
    return _FONT[key]


def render_math(latex, fontsize=17):
    key = (latex, fontsize)
    if key in _MATH:
        return _MATH[key]
    buf = BytesIO()
    math_to_image(
        f"${latex}$",
        buf,
        prop=FontProperties(size=fontsize),
        dpi=150,
        color=INK_HEX,
    )
    im = Image.open(buf).convert("RGBA")
    arr = np.array(im)
    white = arr[:, :, :3].min(axis=2)
    alpha = np.clip((248 - white.astype(np.int16)) * 5, 0, 255).astype(np.uint8)
    out = np.zeros_like(arr)
    out[:, :, 0] = INK[0]
    out[:, :, 1] = INK[1]
    out[:, :, 2] = INK[2]
    out[:, :, 3] = alpha
    im = Image.fromarray(out, "RGBA")
    box = im.getbbox()
    if box:
        im = im.crop(box)
    if im.width > CONTENT_W - 20:
        scale = (CONTENT_W - 20) / im.width
        im = im.resize((int(im.width * scale), max(1, int(im.height * scale))), Image.Resampling.LANCZOS)
    _MATH[key] = im
    return im


class Notes:
    def __init__(self):
        self.pages = []
        self.page_no = 0
        self.line_i = 0
        self.im = None
        self.dr = None
        self.new_page(first=True)

    def new_page(self, first=False):
        im = Image.new("RGB", (W, H), PAPER)
        self.im = im
        self.dr = ImageDraw.Draw(im)
        self.pages.append(im)
        self.page_no += 1
        self.line_i = 0
        if first:
            self.blank(1)
            self.title("Assignment - 2", 72)
            self.title("CHL-7201", 64)
            self.blank(1)

    def ensure(self, n):
        if LINE0 + (self.line_i + n) * LINE_H > BOTTOM:
            self.new_page()

    def blank(self, n=1):
        self.ensure(n)
        self.line_i += n

    def _draw_line(self, text, size, indent, bold, center):
        self.ensure(1)
        face = font(size, bold)
        y = LINE0 + self.line_i * LINE_H
        baseline = y + LINE_H - 14
        if center:
            width = face.getlength(text)
            x = (W - width) / 2
        else:
            x = MARGIN_L + indent
        self.dr.text((x, baseline), text, font=face, fill=INK, anchor="ls")
        if bold:
            width = face.getlength(text)
            self.dr.line([(x, baseline + 6), (x + width, baseline + 6)], fill=INK, width=2)
        self.line_i += 1

    def wrap(self, text, size, indent):
        face = font(size, False)
        width = CONTENT_W - indent
        lines, cur = [], ""
        for word in text.split():
            trial = word if not cur else cur + " " + word
            if face.getlength(trial) <= width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines or [""]

    def text(self, text, size=32, indent=0):
        for line in self.wrap(text, size, indent):
            self._draw_line(line, size, indent, False, False)

    def title(self, text, size):
        self.ensure(2)
        face = font(size)
        y = LINE0 + self.line_i * LINE_H
        baseline = y + 2 * LINE_H - 16
        width = face.getlength(text)
        x = (W - width) / 2
        self.dr.text((x, baseline), text, font=face, fill=INK, anchor="ls")
        self.dr.line([(x, baseline + 8), (x + width, baseline + 10)], fill=INK, width=3)
        self.line_i += 2

    def center(self, text, size, bold=False):
        self._draw_line(text, size, 0, bold, True)

    def heading(self, text):
        self.ensure(1)
        self._draw_line(text, 38, 0, True, False)

    def math(self, latex, indent=28, fontsize=19):
        im = render_math(latex, fontsize=fontsize)
        slots = max(1, int(np.ceil((im.height + 8) / LINE_H)))
        self.ensure(slots)
        y = LINE0 + self.line_i * LINE_H
        block = slots * LINE_H
        top = y + max(0, (block - im.height) // 2) - 4
        self.im.paste(im, (MARGIN_L + indent, int(top)), im)
        self.line_i += slots

    def table(self, headers, rows, font_size=24):
        face = font(font_size)
        cols = len(headers)
        widths = []
        for c in range(cols):
            texts = [headers[c]] + [row[c] for row in rows]
            widths.append(max(face.getlength(t) for t in texts) + 22)
        total = sum(widths)
        if total > CONTENT_W - 8:
            font_size = max(18, int(font_size * (CONTENT_W - 8) / total))
            face = font(font_size)
            widths = []
            for c in range(cols):
                texts = [headers[c]] + [row[c] for row in rows]
                widths.append(max(face.getlength(t) for t in texts) + 18)
            scale = (CONTENT_W - 8) / sum(widths)
            widths = [w * scale for w in widths]
        row_h = font_size + 16
        n_rows = 1 + len(rows)
        height = n_rows * row_h + 8
        slots = max(1, int(np.ceil(height / LINE_H)))
        self.ensure(slots)
        x0 = MARGIN_L + 8
        y0 = LINE0 + self.line_i * LINE_H + 2
        y = y0
        all_rows = [headers] + rows
        for r, row in enumerate(all_rows):
            x = x0
            for c, cell in enumerate(row):
                self.dr.rectangle([x, y, x + widths[c], y + row_h], outline=INK, width=2)
                self.dr.text((x + 8, y + row_h - 8), cell, font=face, fill=INK, anchor="ls")
                x += widths[c]
            y += row_h
        self.line_i += slots


def build():
    n = Notes()

    n.heading("Q1. Scalar RNN")
    n.math(r"h_t=\tanh(z_t),\quad z_t=W_{xh}x_t+W_{hh}h_{t-1}+b_h")
    n.math(r"y_t=W_{hy}h_t+b_y,\quad E_t=(d_t-y_t)^2")
    n.text("The training loss is only E = E2 + E3. Here h0 = 0.2.")
    n.blank()

    n.heading("(a) Forward pass")
    n.math(r"\tanh'(z_t)=1-h_t^{2}")
    n.table(
        ["t", "z_t", "h_t", "tanh'", "y_t", "E_t"],
        [
            ["1", "0.6200", "0.5511", "0.6963", "0.6614", "0.0683"],
            ["2", "0.2707", "0.2643", "0.9302", "0.3171", "0.0003"],
            ["3", "0.9786", "0.7524", "0.4338", "0.9029", "0.0106"],
        ],
    )
    n.math(r"z_1=0.8(0.5)+0.6(0.2)+0.1=0.6200")
    n.math(r"E=E_2+E_3=0.0003+0.0106=0.0109")
    n.blank()

    n.heading("(b) Output weight")
    n.math(r"\frac{\partial E_t}{\partial y_t}=-2(d_t-y_t)")
    n.math(r"\frac{\partial E}{\partial W_{hy}}=\sum_{t=2}^{3}\frac{\partial E_t}{\partial y_t}\, h_t")
    n.math(r"\frac{\partial E_2}{\partial W_{hy}}=0.0342\times 0.2643=0.0090")
    n.math(r"\frac{\partial E_3}{\partial W_{hy}}=0.2059\times 0.7524=0.1549")
    n.math(r"\frac{\partial E}{\partial W_{hy}}=0.0090+0.1549=0.1639")
    n.blank()

    n.heading("(c) BPTT for Whh and Wxh")
    n.text("As in the lecture, for each loss time T in {2, 3} and each earlier state i at most T,")
    n.math(r"\frac{\partial E_T}{\partial W}=\sum_{i=1}^{T}\frac{\partial E_T}{\partial h_T}\cdot\frac{\partial h_T}{\partial h_i}\cdot\frac{\partial h_i}{\partial W}_{local}")
    n.math(r"\frac{\partial E_T}{\partial h_T}=-2(d_T-y_T)\, W_{hy}")
    n.math(r"\frac{\partial h_j}{\partial h_{j-1}}=W_{hh}\,\tanh'(z_j)")
    n.math(r"\frac{\partial h_i}{\partial W_{hh}}_{local}=\tanh'(z_i)\, h_{i-1}")
    n.math(r"\frac{\partial h_i}{\partial W_{xh}}_{local}=\tanh'(z_i)\, x_i")
    n.text("Because h0 = 0.2 is not zero, the i = 1 term for Whh is kept. Both E2 and E3 are added.")
    n.math(r"\frac{\partial E_2}{\partial h_2}=0.0411,\quad \frac{\partial E_3}{\partial h_3}=0.2470")
    n.text("Backward chain:")
    n.math(r"\frac{\partial h_2}{\partial h_1}=0.6\times 0.9302=0.5581")
    n.math(r"\frac{\partial h_3}{\partial h_2}=0.6\times 0.4338=0.2603")
    n.math(r"\frac{\partial h_3}{\partial h_1}=0.2603\times 0.5581=0.1453")
    n.text("From E2. The chain column is dh2 / dhi.")
    n.table(
        ["i", "chain", "dh/dWhh", "dh/dWxh", "Whh", "Wxh"],
        [
            ["1", "0.5581", "0.1393", "0.3481", "0.0032", "0.0080"],
            ["2", "1", "0.5126", "-0.1860", "0.0210", "-0.0076"],
        ],
    )
    n.text("From E3. The chain column is dh3 / dhi.")
    n.table(
        ["i", "chain", "dh/dWhh", "dh/dWxh", "Whh", "Wxh"],
        [
            ["1", "0.1453", "0.1393", "0.3481", "0.0050", "0.0125"],
            ["2", "0.2603", "0.5126", "-0.1860", "0.0330", "-0.0120"],
            ["3", "1", "0.1146", "0.3905", "0.0283", "0.0965"],
        ],
    )
    n.math(r"\frac{\partial E}{\partial W_{hh}}=0.0032+0.0210+0.0050+0.0330+0.0283=0.0905")
    n.text("The five Wxh entries above add to 0.0974 after each has been rounded. Before that rounding the sum is 0.0973, and that is the value that matches a finite-difference check. Use 0.0973 for the update. Either value gives the same new Wxh at four decimal places.")
    n.math(r"\frac{\partial E}{\partial W_{xh}}=0.0973,\quad \frac{\partial E}{\partial W_{hh}}=0.0905,\quad \frac{\partial E}{\partial W_{hy}}=0.1639")
    n.blank()

    n.heading("(d) One gradient step, learning rate 0.5")
    n.text("Biases stay as they are. Only the three weights move.")
    n.math(r"W\leftarrow W-\eta\frac{\partial E}{\partial W}")
    n.math(r"W_{xh}:\ 0.8000-0.5(0.0973)=0.7513")
    n.math(r"W_{hh}:\ 0.6000-0.5(0.0905)=0.5547")
    n.math(r"W_{hy}:\ 1.2000-0.5(0.1639)=1.1180")
    n.text("New forward pass: E2 = 0.0012, E3 = 0.0000, so")
    n.math(r"E_{new}=0.0012")
    n.text("The loss fell, from 0.0109 to 0.0012.")
    n.blank()

    n.heading("(e) Vanishing gradients")
    n.text("(i) From the chain above,")
    n.math(r"\frac{\partial h_3}{\partial h_1}=0.1453")
    n.text("(ii) From h1 to h50 there are 49 factors, and each satisfies |Whh| tanh' at most |Whh| = 0.6. So")
    n.math(r"\left|\frac{\partial h_{50}}{\partial h_1}\right|\leq 0.6^{49}=1.3471\times 10^{-11}")
    n.text("(iii) On the GRU additive path the factor per step is 1 - z_t. With z_t = 0.1 for 49 steps,")
    n.math(r"(1-0.1)^{49}=0.9^{49}=0.0057")
    n.text("(iv) Clipping only shrinks a gradient whose norm is already above the threshold. The product")
    n.math(r"1.3\times 10^{-11}")
    n.text("sits far below any threshold, so the cap leaves it that small and the early-step update stays negligible.")
    n.text("Truncated BPTT with chunk length k = 20 carries the hidden state forward through the whole sequence, but backpropagates only inside each chunk of 20 steps. A dependency that runs from step 1 to step 50 crosses a chunk boundary, so the loss at step 50 sends no gradient back to step 1. That long dependency is left untrained.")
    n.blank()

    n.heading("Q2. Encoder-decoder")
    n.math(r"h_t=\tanh(W_x x_t+W_h h_{t-1}+b),\quad h_0=0,\quad c=h_3")
    n.heading("(a) Encoder")
    n.math(r"z_1=(0.5,\ -0.3),\quad h_1=(0.4621,\ -0.2913)")
    n.math(r"h_2=(0.3848,\ -0.2727)")
    n.math(r"h_3=(-0.3569,\ 0.3286)")
    n.math(r"c=h_3=(-0.3569,\ 0.3286)")
    n.blank()

    n.heading("(b) Decoder step 1")
    n.text("s0 = c, and the START embedding is (0, 0). Since Ws = 0.5 I and Wc = I,")
    n.math(r"s_1=\tanh(1.5\, c)=(-0.4895,\ 0.4565)")
    n.text("Logits, in the order Normal, Alarm, END:")
    n.math(r"(-0.3907,\ 1.4024,\ -0.4670)")
    n.math(r"e^{-0.3907}=0.6766,\quad e^{1.4024}=4.0649,\quad e^{-0.4670}=0.6269")
    n.text("The sum is 5.3684, so")
    n.math(r"P=(0.1260,\ 0.7572,\ 0.1168)")
    n.text("The predicted token is Alarm. The true token is Normal.")
    n.blank()

    n.heading("(c) Step 2, two ways")
    n.text("Teacher forcing, feed the true y1* = Normal, e = (1, 0):")
    n.math(r"s_2=(-0.9219,\ -0.4163)")
    n.math(r"P=(0.1991,\ 0.2571,\ 0.5438),\quad \hat{y}_2=\mathrm{END}")
    n.text("Inference, feed the model's own y1 = Alarm, e = (0, 1):")
    n.math(r"s_2=(-0.9219,\ 0.7844)")
    n.math(r"P=(0.0196,\ 0.9270,\ 0.0534),\quad \hat{y}_2=\mathrm{Alarm}")
    n.text("During training the decoder is shown the true previous token, so step 2 is computed from Normal and it does emit END. At inference the wrong token Alarm is written back in, the second state is different, and the decoder emits Alarm again. If decoding stops at END or after 2 steps, inference produces (Alarm, Alarm). END has not appeared, so the run stops because of the 2-step limit.")
    n.blank()

    n.heading("Q3. Attention")
    n.text("The score of each encoder state is the dot product of the previous decoder state with that state. Softmax turns the scores into weights, and the context is their weighted sum. The decoder equation is the same as in Q2, with this new context.")
    n.math(r"e_{ti}=s_{t-1}^{T}h_i,\quad c_t=\sum_i\alpha_{ti}h_i")
    n.blank()

    n.heading("(a) Step 1, s0 = h3")
    n.math(r"e_1=(-0.2606,\ -0.2270,\ 0.2353)")
    n.math(r"\alpha_1=(0.2720,\ 0.2813,\ 0.4467)")
    n.math(r"c_1=(0.0745,\ -0.0092)")
    n.math(r"s_1=(-0.1035,\ 0.1538)")
    n.math(r"P=(0.4393,\ 0.4056,\ 0.1551)")
    n.text("y1 = Normal.")
    n.blank()

    n.heading("(b) Step 2, teacher forcing with Normal")
    n.math(r"e_2=(-0.0927,\ -0.0818,\ 0.0875)")
    n.math(r"\alpha_2=(0.3117,\ 0.3151,\ 0.3732)")
    n.math(r"c_2=(0.1321,\ -0.0541)")
    n.math(r"s_2=(-0.7257,\ -0.7518)")
    n.math(r"P=(0.3377,\ 0.0976,\ 0.5648)")
    n.text("y2 = END. The predicted sequence is (Normal, END), which is the true sequence.")
    n.blank()

    n.heading("(c) The two weight vectors")
    n.text("Step 1 puts the most weight on h3 (0.4467). Step 2 still puts the most weight on h3 (0.3732), with the three weights closer together. The scores depend on the previous decoder state. When that state changes, the softmax changes, and the weighted sum of h1, h2 and h3 changes with it. That is why the context is rebuilt at every step.")
    n.blank()

    n.heading("(d) Mean pooling, each weight = 1/3")
    n.math(r"c=(0.1633,\ -0.0785)")
    n.math(r"s_1=(-0.0151,\ 0.0856)")
    n.math(r"P=(0.5302,\ 0.3198,\ 0.1500)")
    n.text("(i) P(Normal) is 0.1260 when c = h3 (Q2), 0.4393 with attention, and 0.5302 with equal weights.")
    n.text("(ii) The values 1/3 are fixed. They do not depend on the previous decoder state, or on which encoder state matters. Training changes the encoder and decoder weights, so the hidden states change. The dot products then grow for the states that help the token being predicted, the weights move away from 1/3, and the pattern differs from one decoding step to the next.")
    n.blank()

    n.heading("Q4. VAE")
    n.math(r"\mu=w_{\mu}^{T}x+b_{\mu},\quad \log\sigma^{2}=w_{v}^{T}x+b_{v}")
    n.math(r"z=\mu+\sigma\varepsilon,\quad \hat{x}=\mathrm{sigmoid}(W'z+b')")
    n.math(r"L=\sum_{j=1}^{2}(x_j-\hat{x}_j)^{2}+\frac{1}{2}(\mu^{2}+\sigma^{2}-\log\sigma^{2}-1)")
    n.blank()

    n.heading("(a) Encoder")
    n.math(r"\mu=0.5(0.8)+0.5(0.4)=0.6000")
    n.math(r"\log\sigma^{2}=-1(0.8)+0.5(0.4)-0.6=-1.2000")
    n.math(r"\sigma^{2}=e^{-1.2}=0.3012,\quad \sigma=e^{-0.6}=0.5488")
    n.math(r"z=0.6000+0.5488(0.5)=0.8744")
    n.blank()

    n.heading("(b) Decoder and loss")
    n.math(r"u=(1.2488,\ -0.1256),\quad \hat{x}=(0.7771,\ 0.4686)")
    n.math(r"L_{rec}=(0.8-0.7771)^{2}+(0.4-0.4686)^{2}=0.0052")
    n.math(r"L_{KL}=\frac{1}{2}\left(0.6000^{2}+0.3012-(-1.2000)-1\right)=0.4306")
    n.math(r"L=0.0052+0.4306=0.4358")
    n.blank()

    n.heading("(c) Gradients through the reparameterisation")
    n.text("Let l stand for log of sigma squared, so sigma = e to the power l/2, and")
    n.math(r"\frac{\partial z}{\partial \ell}=\frac{\varepsilon\sigma}{2}")
    n.math(r"\frac{\partial L_{rec}}{\partial \hat{x}_j}=-2(x_j-\hat{x}_j)")
    n.math(r"\frac{\partial \hat{x}_j}{\partial u_j}=\hat{x}_j(1-\hat{x}_j)")
    n.math(r"\frac{\partial L_{rec}}{\partial z}=0.0183")
    n.text("The KL term contributes mu for the mean, and (sigma squared minus 1)/2 for l:")
    n.math(r"\frac{\partial L}{\partial \mu}=0.0183+0.6000=0.6183")
    n.math(r"\frac{\partial L}{\partial \ell}=0.0183\cdot\frac{0.5\times 0.5488}{2}+\frac{0.3012-1}{2}=-0.3469")
    n.text("Sampling z straight from N(mu, sigma^2) is a random node. It has no derivative, so the chain from L stops there and the encoder never receives a gradient. Writing z = mu + sigma epsilon, with epsilon drawn outside the graph, turns that step into addition and multiplication, which do have derivatives.")
    n.blank()

    n.heading("(d) Monitoring. Noise is 0, so z = mu")
    n.text("Normal sample x = (0.8, 0.4), and mu = 0.6000.")
    n.math(r"\hat{x}=(0.6682,\ 0.4013),\quad \mathrm{error}=0.0174")
    n.text("Suspect sample xf = (0.2, 0.9), and mu = 0.5500.")
    n.math(r"\hat{x}=(0.6457,\ 0.3894),\quad \mathrm{error}=0.4594")
    n.text("The suspect sample is the one to flag. Its reconstruction error is 0.4594 / 0.0174 = 26.44 times the normal sample's error.")
    n.blank()

    n.heading("(e) A new sample, z = -1")
    n.math(r"u=(-2.5,\ -2),\quad \hat{x}=(0.0759,\ 0.1192)")
    n.text("The KL term pulls q(z|x) toward N(0, 1), so the decoder is trained on codes spread over that prior. Drawing z = -1 from the prior and decoding is a draw from the learned model. A plain autoencoder stores one point per input and has no prior, so there is no distribution to draw z from.")

    pages = []
    rng = np.random.default_rng(7)
    for im in n.pages:
        arr = np.array(im).astype(np.int16)
        noise = rng.integers(-3, 4, size=arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        pages.append(Image.fromarray(arr, "RGB"))
    pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=DPI)
    print(f"wrote {OUT} ({len(pages)} pages)")


if __name__ == "__main__":
    build()
