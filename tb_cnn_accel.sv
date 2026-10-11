`timescale 1ns/1ps
module tb_cnn_accel;
    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic [3:0] pred_digit;

    cnn_accel_top dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .busy       (busy),
        .done       (done),
        .pred_digit (pred_digit)
    );

    initial clk = 1'b0;
    always #10 clk = ~clk; // 50 MHz equivalent period = 20 ns

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        #100;
        rst_n = 1'b1;
        #100;
        start = 1'b1;
        #20;
        start = 1'b0;

        wait(done == 1'b1);
        $display("[TB] Inference done. Predicted digit = %0d", pred_digit);
        #100;
        $finish;
    end
endmodule
