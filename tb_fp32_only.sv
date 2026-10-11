`timescale 1ns/1ps
// ============================================================
//  tb_fp32_only.sv
//  Testbench CHI test DUT FP32 (cnn_accel_fp32 — shortreal, khong
//  luong tu hoa). Bo het INT8/INT4/FP16/Pipeline de chay NHANH hon
//  nhieu so voi tb_compare_all_fp32.sv.
//
//  In ket qua dang:
//    [CMP] image_<d>_<r> label=<l> fp32=<p>  (OK/SAI)
//  va cuoi cung in bang accuracy theo tung chu so + latency.
//
//  CACH CHAY THU NHANH: doi N_PER_DIGIT xuong 10 (=> 100 anh thay vi
//  1000) de kiem tra luong chay truoc, sau do dat lai 100.
// ============================================================
module tb_fp32_only;

    // So anh moi chu so (100 => 1000 anh nhu ban goc; giam de chay thu)
    localparam int N_PER_DIGIT = 100;

    // ── Clock & reset ─────────────────────────────────────────────────
    logic clk, rst_n, start;
    initial clk = 0;
    always  #10 clk = ~clk;   // chu ky 20ns

    // ── FP32 DUT (shortreal, khong luong tu hoa) ──────────────────────
    logic       busy_f32, done_f32;
    logic [3:0] pred_f32;
    cnn_accel_fp32 u_fp32 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy_f32), .done(done_f32), .pred_digit(pred_f32)
    );

    // ── Counters & bookkeeping ────────────────────────────────────────
    int digit, rank, label;
    string fname_f32;

    int total;
    int cor_f32 [0:9];
    int tot_d   [0:9];

    longint cyc_f32_total;
    longint cyc_start;

    initial begin
        // Khoi tao
        total = 0; cyc_f32_total = 0;
        for (int d = 0; d < 10; d++) begin
            cor_f32[d] = 0; tot_d[d] = 0;
        end

        // Global reset
        rst_n = 0; start = 0;
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5)  @(posedge clk);

        // ── Vong lap chinh: N_PER_DIGIT anh moi chu so ────────────────
        for (digit = 0; digit < 10; digit++) begin
            for (rank = 0; rank < N_PER_DIGIT; rank++) begin
                label = digit;

                // Ten file (2 chu so, giu dung cach dat ten ban goc)
                if (rank < 10)
                    $sformat(fname_f32, "mem/fp32/batch/image_%0d_0%0d.hex", digit, rank);
                else
                    $sformat(fname_f32, "mem/fp32/batch/image_%0d_%0d.hex",  digit, rank);

                // Nap anh (bit pattern IEEE-754 32-bit moi pixel)
                $readmemh(fname_f32, u_fp32.image_mem);

                // Reset ngan giua cac anh
                rst_n = 0;
                repeat(4) @(posedge clk);
                rst_n = 1;
                repeat(2) @(posedge clk);

                // Ghi cycle bat dau
                cyc_start = $time / 20; // 20ns / cycle

                // Start
                start = 1;
                @(posedge clk);
                start = 0;

                // Cho FP32 done
                wait(done_f32 == 1);
                cyc_f32_total += ($time/20 - cyc_start);

                @(posedge clk);

                // Ghi ket qua
                tot_d[label]++;
                total++;
                if (pred_f32 == label) cor_f32[label]++;

                $display("[CMP] image_%0d_%02d label=%0d fp32=%0d  %s",
                    digit, rank, label, pred_f32,
                    (pred_f32 == label) ? "OK" : "SAI");

                // Cho done ha
                wait(done_f32 == 0);
                repeat(2) @(posedge clk);
            end
        end

        // ── Bao cao ───────────────────────────────────────────────────
        $display("");
        $display("================================================================================");
        $display("  DO CHINH XAC — FP32 (shortreal, khong luong tu hoa)");
        $display("  Tong so mau: %0d (%0d mau x 10 chu so)", total, N_PER_DIGIT);
        $display("================================================================================");
        $display("  Digit | FP32(acc) | dung/tong");
        $display("--------|-----------|-----------");
        for (int d = 0; d < 10; d++) begin
            $display("    %0d   | %6.2f%%   | %0d/%0d",
                d, 100.0*cor_f32[d]/tot_d[d], cor_f32[d], tot_d[d]);
        end
        $display("--------|-----------|-----------");

        // Tong dung (khai bao automatic de tranh vlog-2244)
        begin
            automatic int s32 = 0;
            for (int d = 0; d < 10; d++) s32 += cor_f32[d];
            $display("  TOTAL | %6.2f%%   | %0d/%0d",
                100.0*s32/total, s32, total);
        end
        $display("================================================================================");

        // Latency
        $display("");
        $display("  LATENCY TRUNG BINH MOI ANH (cycles):");
        $display("  FP32(sim): %0d cycles  (*)", cyc_f32_total/total);
        $display("  (*) FP32 la mo hinh hanh vi don gian (khong pipeline fetch/mac),");
        $display("      so cycle CHI mang tinh tham khao — KHONG dai dien cho hieu");
        $display("      nang phan cung that (vi module nay khong tong hop duoc).");
        $display("================================================================================");

        #100;
        $finish;
    end
endmodule
