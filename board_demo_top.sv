module board_demo_top (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       start_btn,
    input  logic [3:0] sw,          // (khong con dung) truoc day: chon anh ROM
    input  logic       sw_draw,     // (khong con dung) truoc day: chon che do
    output logic       done_led,
    output logic [3:0] digit_bin,
    output logic [6:0] seg0,
    output logic [6:0] seg1
);
    logic busy;
    logic done;
    logic [3:0] pred_digit;

    // Bo tang toc CHI DUNG draw_ram (anh ve tay nap qua In-System Memory Editor)
    cnn_accel_opt u_accel (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (~start_btn),   // KEY active-low -> dao thanh active-high
        .busy       (busy),
        .done       (done),
        .pred_digit (pred_digit)
    );

    assign done_led  = done;
    assign digit_bin = pred_digit;

    // HEX0: hien chu so CNN du doan duoc
    sevenseg u_seg0 (
        .value(pred_digit),
        .seg  (seg0)
    );

    // HEX1: DA TAT -- value=4'hF de sevenseg tat toan bo doan (blank)
    sevenseg u_seg1 (
        .value(4'hF),
        .seg  (seg1)
    );
    // Ghi chu: sw, sw_draw khong con duoc dung (giu trong cong de khong phai sua .qsf).
    // Quartus co the bao "warning: input never used" -- vo hai.
endmodule
