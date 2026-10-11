`timescale 1ns/1ps
//==============================================================
//  tb_opt_cycles.sv
//  Dem so chu ky theo tung giai doan - BAN TOI UU (cnn_accel_opt)
//  Ma state (theo GIA TRI SO):
//    Conv = 1,2,3,4 | Pool = 5,6,7,8 | FC = 9,10,11,12 | ArgMax = 13,14 | done = 15
//  Ban opt doc anh truc tiep tu draw_ram (khong co pha Load rieng).
//  Noi dung anh KHONG anh huong so chu ky.
//  Compile:  vlog -sv rtl/draw_ram_sim.v rtl/cnn_accel_opt.sv tb/tb_opt_cycles.sv
//  Run:      vsim -c tb_opt_cycles -do "run -all; quit"
//==============================================================
module tb_opt_cycles;
    logic       clk = 0;
    logic       rst_n = 0;
    logic       start = 0;
    logic       busy, done;
    logic [3:0] pred_digit;

    cnn_accel_opt dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .busy       (busy),
        .done       (done),
        .pred_digit (pred_digit)
    );

    always #10 clk = ~clk;   // chu ky 20 ns = 50 MHz

    integer o_conv, o_pool, o_fc, o_arg, cyc;

    function integer phase_of(input [3:0] st);
      case (st)
        4'd1,  4'd2,  4'd3,  4'd4:    phase_of = 1;  // Conv
        4'd5,  4'd6,  4'd7,  4'd8:    phase_of = 2;  // Pool
        4'd9,  4'd10, 4'd11, 4'd12:   phase_of = 3;  // FC
        4'd13, 4'd14:                 phase_of = 4;  // ArgMax
        default:                      phase_of = 0;  // Idle/done
      endcase
    endfunction

    always @(posedge clk) begin
      if (start) begin
        cyc<=0; o_conv<=0; o_pool<=0; o_fc<=0; o_arg<=0;
      end else if (!done) begin
        cyc <= cyc + 1;
        case (phase_of(dut.state))    // neu ModelSim bao loi enum: doi thanh int'(dut.state)
          1: o_conv <= o_conv + 1;
          2: o_pool <= o_pool + 1;
          3: o_fc   <= o_fc   + 1;
          4: o_arg  <= o_arg  + 1;
        endcase
      end
    end

    always @(posedge done)
      $display("OPT   Conv=%0d  Pool=%0d  FC=%0d  ArgMax=%0d  |  Tong(pha)=%0d  cyc=%0d",
               o_conv, o_pool, o_fc, o_arg,
               o_conv+o_pool+o_fc+o_arg, cyc);

    initial begin
      rst_n = 0; start = 0;
      repeat(5) @(posedge clk);
      rst_n = 1;
      repeat(5) @(posedge clk);
      @(posedge clk); start <= 1;   // pulse start 1 chu ky
      @(posedge clk); start <= 0;
      wait(done);
      repeat(3) @(posedge clk);
      $display("OPT: xong.");
      $finish;
    end
endmodule
