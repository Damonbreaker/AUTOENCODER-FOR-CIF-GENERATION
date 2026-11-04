#!/usr/bin/env python3
"""Submission report for CHL7201 Assignment 2, Problem 5.

Numbers are the ones printed by CHL7201_Assignment2.ipynb.
"""

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "CHL7201_Problem5_Report.pdf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

DISCUSSION = (
    "Deploy the vanilla RNN soft sensor at L = 6 (test RMSE 1.18; training-mean "
    "baseline 1.21). The L = 12 LSTM (1.23) and GRU (1.23) are higher. From L = 6 "
    "to L = 48 the RNN error rises to 1.20, since each backward step is scaled by "
    "at most |W_hh| tanh' and distant hours fade. The RNN gradient norm starts at "
    "1.9, so the clip of 1 binds; the LSTM starts at 0.7. Deploy the attention "
    "encoder-decoder as the 6-hour forecaster. Inference RMSE at 6 hours is 1.20, "
    "versus 1.25 with no attention and 1.31 for repeating the last silica. Teacher "
    "forcing stays between 0.73 and 0.86 while inference rises from 0.86 to 1.25, "
    "as each prediction is fed back. Plotted weights peak 10-19 hours back. A VAE "
    "with epsilon = 0 on the 21 process variables should flag hours whose "
    "reconstruction error is many times the normal value."
)


class Report(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font("DejaVu", size=8)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, f"CHL7201 / CLL788  ·  Problem 5 report  ·  {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def main():
    pdf = Report(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_font("DejaVu", "", FONT)
    pdf.add_font("DejaVu", "B", FONTB)
    pdf.set_margin(16)
    pdf.add_page()

    def h1(text):
        pdf.set_font("DejaVu", "B", 13)
        pdf.multi_cell(0, 6.4, text, align="C")
        pdf.ln(1.5)

    def h2(text):
        pdf.ln(2.5)
        pdf.set_font("DejaVu", "B", 12)
        pdf.multi_cell(0, 6.2, text)
        pdf.ln(1)

    def body(text):
        pdf.set_font("DejaVu", size=10.5)
        pdf.multi_cell(0, 5.3, text)
        pdf.ln(1.4)

    def caption(text):
        pdf.set_font("DejaVu", size=9)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 4.4, text)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1.6)

    def table(headers, rows, widths):
        pdf.set_font("DejaVu", "B", 9)
        pdf.set_fill_color(232, 236, 242)
        for head, width in zip(headers, widths):
            pdf.cell(width, 7, head, border=1, align="C", fill=True)
        pdf.ln()
        pdf.set_font("DejaVu", size=9)
        for row in rows:
            for value, width in zip(row, widths):
                pdf.cell(width, 6.4, value, border=1, align="C")
            pdf.ln()
        pdf.ln(2)

    def figure(name, cap, width=178):
        path = FIG / name
        if pdf.get_y() + 70 > pdf.h - pdf.b_margin:
            pdf.add_page()
        pdf.image(str(path), w=width)
        pdf.ln(1.2)
        caption(cap)

    h1("CHL7201/CLL788: Assignment 2 - RNN, Encoder-Decoder, Attention and VAE")
    pdf.ln(1)
    h1("Report for Problem 5")
    pdf.ln(1)
    body(
        "This report describes the coding experiments. All networks were implemented "
        "in PyTorch. Random seed = 0."
    )
    body(
        "The notebook sets both torch.manual_seed(0) and np.random.seed(0) before "
        "training, and every table below is taken from that run. The code is "
        "CHL7201_Assignment2.ipynb, section Q5. Loss during training is mean squared "
        "error on the scaled target. Every RMSE below is converted back to the "
        "original units, percent silica."
    )

    h2("1. Data preparation")
    body(
        "The file is the flotation-plant database. It has 737,453 rows and 24 columns, "
        "from 10 March 2017, 01:00, to 9 September 2017, 23:00. The numbers use a "
        "decimal comma, so the file is read with decimal=','. A typical hour contains "
        "180 rows, one sample every 20 seconds. Averaging inside each clock hour "
        "leaves 4,097 hours."
    )
    body(
        "% Iron Concentrate is a second laboratory assay of the same product as "
        "% Silica Concentrate. It is dropped before any model sees the table, so the "
        "network cannot read the answer off a paired lab measurement. That leaves "
        "21 process columns. Silica is the target. In the soft sensor, silica is not "
        "an input. In the forecaster, past silica is an input, because the task is "
        "to continue the silica series."
    )
    body(
        "There is one break in the clock. 16 March 2017, 05:00 jumps to 29 March "
        "2017, 12:00, a gap of 13 days and 7 hours. Hours are grouped into contiguous "
        "segments, and any window that would step across that shutdown is discarded. "
        "The same rule keeps a window from mixing training, validation and test hours."
    )
    body(
        "The hours are cut in time order, 70/15/15, which is 2,867 / 614 / 616 hours. "
        "The scaler is fit on the training hours only. On those hours the silica mean "
        "is 2.3176 and the standard deviation is 1.1256. Predicting that training "
        "mean on the L = 12 test windows gives RMSE 1.2108. That is the baseline "
        "the soft sensors have to beat."
    )

    h2("2. Soft sensor")
    body(
        "The soft sensor predicts silica at the next hour from the previous L hours "
        "of the 21 process variables. Three models share the same shape: one recurrent "
        "layer with 32 units, then a linear output. The vanilla RNN uses tanh. "
        "Optimiser is Adam with learning rate 1e-4, for 20 epochs, batch size 128. "
        "Gradients are clipped at norm 1. The same budget is used for every model "
        "and every window length."
    )
    body("At L = 12 the test errors are:")
    table(
        ["Model", "Parameters", "Test RMSE"],
        [
            ["Vanilla RNN", "1,793", "1.1837"],
            ["LSTM", "7,073", "1.2264"],
            ["GRU", "5,313", "1.2333"],
            ["Training-mean baseline", "—", "1.2108"],
        ],
        [70, 54, 54],
    )
    body(
        "The vanilla RNN is the only one of the three that beats the baseline at "
        "this window. The LSTM and the GRU have several times as many parameters "
        "and a higher test error. Extra gating is not helping this one-hour target."
    )
    figure(
        "partB_loss.png",
        "Figure 1. Training loss (solid) and validation loss (dashed) at L = 12, "
        "in scaled silica units. Training loss falls for all three models. "
        "Validation loss flattens. Twenty epochs is enough to see the comparison, "
        "and it is the same number of epochs for each model.",
    )
    figure(
        "partB_gradnorm.png",
        "Figure 2. Mean gradient norm before the clip, at L = 12. The dotted line "
        "is the clip threshold of 1.",
    )
    body(
        "At epoch 1 the RNN norm is 1.907, so the clip of 1 binds and the step is "
        "cut down. By epoch 20 the RNN norm is 0.628 and the clip no longer binds. "
        "The LSTM starts at 0.658 and ends at 0.459, under the clip the whole way. "
        "Clipping can stop a gradient that has already grown too large. It cannot "
        "bring back a gradient that has vanished along a long chain. The next plot "
        "is where that shows up."
    )
    body(
        "The same two models were retrained at L = 6, 24 and 48, with the same "
        "hidden size, learning rate, epochs and clip."
    )
    table(
        ["L (hours)", "RNN test RMSE", "LSTM test RMSE"],
        [
            ["6", "1.1790", "1.2140"],
            ["12", "1.1837", "1.2264"],
            ["24", "1.1919", "1.2392"],
            ["48", "1.1976", "1.2562"],
        ],
        [59, 59, 60],
    )
    figure(
        "partB_rmse_vs_L.png",
        "Figure 3. Test RMSE against the length of the input window.",
    )
    body(
        "A longer memory does not lower the error. From L = 6 to L = 48 the RNN "
        "moves from 1.1790 to 1.1976, and the LSTM moves from 1.2140 to 1.2562. "
        "Each extra RNN step multiplies the backward signal by at most |W_hh| tanh', "
        "which is smaller than 1, so an hour from two days ago arrives with almost "
        "no gradient. On this plant the useful signal for the next silica reading "
        "is recent. The soft sensor to deploy is the vanilla RNN with L = 6 "
        "(test RMSE 1.1790)."
    )

    h2("3. Six-hour forecast")
    body(
        "The forecaster reads the previous 24 hours of the 21 process variables "
        "together with the past silica, so each input step has 22 numbers. The "
        "target is the next 6 silica values. A window that would cross the March "
        "shutdown, or that would mix the train, validation and test blocks, is dropped."
    )
    body(
        "The encoder is an LSTM with 32 units. Its final state (h_T, c_T) is the "
        "context, and it is also the decoder's starting state. The decoder is an "
        "LSTM that reads one silica number and writes one prediction. Its first "
        "input is the last silica that was actually measured. Under teacher forcing, "
        "every later input is the true previous silica. Under inference, every later "
        "input is the decoder's own previous prediction."
    )
    body(
        "Both forecast models use Adam with learning rate 1e-3, 10 epochs, batch "
        "size 128, and the same clip of 1. The plain encoder-decoder has 11,681 "
        "parameters. Its training MSE goes from 0.823 to 0.371 and its validation "
        "MSE from 0.747 to 0.365, so at 10 epochs the two curves are still together."
    )
    body("Test RMSE at each hour ahead:")
    table(
        ["Hour", "Teacher forcing", "Inference", "Attention", "Repeat last"],
        [
            ["1", "0.8641", "0.8641", "0.8416", "0.7566"],
            ["2", "0.7907", "0.9933", "0.9511", "0.9255"],
            ["3", "0.7640", "1.1059", "1.0422", "1.0511"],
            ["4", "0.7482", "1.1889", "1.1178", "1.1668"],
            ["5", "0.7383", "1.2342", "1.1672", "1.2473"],
            ["6", "0.7324", "1.2541", "1.1970", "1.3055"],
        ],
        [28, 38, 34, 36, 42],
    )
    figure(
        "forecast_rmse.png",
        "Figure 4. Test RMSE against the forecast horizon. Teacher forcing stays "
        "low. Inference rises. Attention inference rises more slowly, and from "
        "hour 3 it is also below repeating the last silica.",
    )
    body(
        "At one hour ahead, teacher forcing and inference are the same number, "
        "0.8641. Both of them start from the measured silica, so there is no "
        "earlier prediction to feed back yet. From hour 2 they separate, and the "
        "gap is larger at hour 6 than at hour 2. Teacher forcing is allowed to see "
        "the true silica at every step, so by hour 6 the decoder has been shown "
        "five true values and the prediction is still a one-step prediction from "
        "a correct history. Inference writes the hour-1 prediction back in as the "
        "input for hour 2. If hour 1 is a bit wrong, hour 2 starts from the wrong "
        "place, and hour 3 starts from a place that is wrong for two reasons. "
        "The error piles up."
    )
    body(
        "Repeating the last measured silica is stronger than either network at "
        "one hour (0.7566) and weaker than both networks by six hours (1.3055). "
        "A one-hour forecast on this plant is close to persistence. A six-hour "
        "forecast is not."
    )

    h2("4. Attention")
    body(
        "The attention model uses the same encoder, the same horizon, and the same "
        "training budget. Before each decoder step the score of encoder state i is "
        "the dot product of the current decoder state with that encoder state. A "
        "softmax turns the scores into weights that add up to 1, and the context "
        "is the weighted sum of the 24 encoder states. A matrix W_c sends the "
        "32-dimensional context down to 8 numbers, and those 8 numbers are joined "
        "to the one silica input before they enter the decoder. That keeps the "
        "laboratory value from being drowned by the context vector. The model has "
        "12,961 parameters. Training MSE goes from 0.773 to 0.371 and validation "
        "MSE from 0.738 to 0.338."
    )
    body(
        "Under inference, attention is below the plain encoder-decoder at every "
        "horizon: 0.8416, 0.9511, 1.0422, 1.1178, 1.1672, 1.1970. From hour 3 it "
        "is also below repeating the last silica. The six-hour forecaster to deploy "
        "is this attention model. Its inference RMSE at 6 hours is 1.1970, against "
        "1.2541 with no attention and 1.3055 for persistence."
    )
    figure(
        "attention_heatmap.png",
        "Figure 5. Attention weights on one test window, under inference. Darker "
        "means a larger weight. The horizontal axis is hours before the last "
        "observation. An equal share of the 24 hours would be 1/24 = 0.0417.",
    )
    body(
        "On this window the largest weight is 19 hours before the last observation "
        "for forecast steps 1, 2 and 3, then 13, 12 and 10 hours before it for "
        "steps 4, 5 and 6. The weights are spread out rather than piled on one "
        "hour. The mean weight on the most recent hour is 0.0445, and the mean "
        "weight on the oldest hour is 0.031. The window leans on a band about "
        "10 to 19 hours back, and that band shifts toward the present as the "
        "forecast steps forward. It does not put all of its weight on the single "
        "most recent hour."
    )

    h2("5. What to deploy")
    body(
        "The discussion below is 150 words. It is the paragraph written by the "
        "notebook from the tables in this report."
    )
    pdf.set_font("DejaVu", size=10.5)
    pdf.set_fill_color(244, 246, 250)
    pdf.multi_cell(0, 5.3, DISCUSSION, fill=True)
    pdf.ln(1)

    pdf.output(str(OUT))
    print(f"wrote {OUT}  pages={pdf.pages_count}  words={len(DISCUSSION.split())}")


if __name__ == "__main__":
    main()
