module board_demo_top (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       start_btn,   // KEY (active-low)
    output logic       done_led,
    output logic [3:0] digit_bin,
    output logic [6:0] seg0,
    output logic [6:0] seg1
);
    logic busy;
    logic done;
    logic [3:0] pred_digit;

    // Bo tang toc GOC (luon nap anh tu draw_ram)
    cnn_accel_top u_accel (
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

    // HEX1: TAT -- value=4'hF de sevenseg tat toan bo doan (blank)
    sevenseg u_seg1 (
        .value(4'hF),
        .seg  (seg1)
    );
endmodule