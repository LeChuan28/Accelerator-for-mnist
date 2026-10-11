`timescale 1ns/1ps
//==============================================================
//  tb_base_cycles.sv
//  Dem so chu ky theo tung giai doan - BAN GOC (cnn_accel_top)
//  Ma state (theo GIA TRI SO, khong dung ten localparam cua DUT):
//    Load anh = 1,2 | Conv = 3,4,5 | Pool = 6,7,8 | FC = 9,10,11 | ArgMax = 12,13,14
//  Luu y: noi dung anh KHONG anh huong so chu ky (vong lap co kich thuoc co dinh),
//  nen khong can nap anh MNIST that. Chi can pulse start va cho toi done.
//  Compile:  vlog -sv rtl/draw_ram_sim.v rtl/cnn_accel_top.sv tb/tb_base_cycles.sv
//  Run:      vsim -c tb_base_cycles -do "run -all; quit"
//==============================================================
module tb_base_cycles;
    logic       clk = 0;
    logic       rst_n = 0;
    logic       start = 0;
    logic       use_draw = 0;   // 0 = doc image_mem (khong quan trong voi dem chu ky)
    logic       busy, done;
    logic [3:0] pred_digit;

    cnn_accel_top dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .use_draw   (use_draw),
        .busy       (busy),
        .done       (done),
        .pred_digit (pred_digit)
    );

    always #10 clk = ~clk;   // chu ky 20 ns = 50 MHz

    integer t_load, t_conv, t_pool, t_fc, t_arg, cyc;

    // Tra ma pha theo gia tri so cua tin hieu state
    function integer phase_of(input [3:0] st);
      case (st)
        4'd1,  4'd2:           phase_of = 0;  // nap anh
        4'd3,  4'd4,  4'd5:    phase_of = 1;  // Conv
        4'd6,  4'd7,  4'd8:    phase_of = 2;  // Pool
        4'd9,  4'd10, 4'd11:   phase_of = 3;  // FC
        4'd12, 4'd13, 4'd14:   phase_of = 4;  // ArgMax
        default:               phase_of = 5;  // Idle/khac
      endcase
    endfunction

    always @(posedge clk) begin
      if (start) begin
        cyc<=0; t_load<=0; t_conv<=0; t_pool<=0; t_fc<=0; t_arg<=0;
      end else if (!done) begin
        cyc <= cyc + 1;
        case (phase_of(dut.state))    // neu ModelSim bao loi enum: doi thanh int'(dut.state)
          0: t_load <= t_load + 1;
          1: t_conv <= t_conv + 1;
          2: t_pool <= t_pool + 1;
          3: t_fc   <= t_fc   + 1;
          4: t_arg  <= t_arg  + 1;
        endcase
      end
    end

    always @(posedge done)
      $display("BASE  Load=%0d  Conv=%0d  Pool=%0d  FC=%0d  ArgMax=%0d  |  Tong(pha)=%0d  cyc=%0d",
               t_load, t_conv, t_pool, t_fc, t_arg,
               t_load+t_conv+t_pool+t_fc+t_arg, cyc);

    initial begin
      rst_n = 0; start = 0; use_draw = 0;
      repeat(5) @(posedge clk);
      rst_n = 1;
      repeat(5) @(posedge clk);
      @(posedge clk); start <= 1;   // pulse start 1 chu ky
      @(posedge clk); start <= 0;
      wait(done);
      repeat(3) @(posedge clk);
      $display("BASE: xong.");
      $finish;
    end
endmodule
