`timescale 1ns/1ps
// ============================================================
//  tb_compare_all.sv
//  Testbench chạy song song 3 DUT: INT8, INT4, FP16
//  Mỗi ảnh: reset cả 3 → start cùng lúc → chờ done cả 3
//  In kết quả dạng:
//  [CMP] image_<d>_<r> label=<l> int8=<p> int4=<p> fp16=<p>
// ============================================================
module tb_compare_all;

    // ── Clock & reset ─────────────────────────────────────────────────
    logic clk, rst_n, start;
    initial clk = 0;
    always  #10 clk = ~clk;

    // ── INT8 DUT ──────────────────────────────────────────────────────
    logic       busy_i8, done_i8;
    logic [3:0] pred_i8;
    cnn_accel_top u_int8 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy_i8), .done(done_i8), .pred_digit(pred_i8)
    );

    // ── INT4 DUT ──────────────────────────────────────────────────────
    logic       busy_i4, done_i4;
    logic [3:0] pred_i4;
    cnn_accel_int4 u_int4 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy_i4), .done(done_i4), .pred_digit(pred_i4)
    );

    // ── FP16 DUT ──────────────────────────────────────────────────────
    logic       busy_f16, done_f16;
    logic [3:0] pred_f16;
    cnn_accel_fp16 u_fp16 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy_f16), .done(done_f16), .pred_digit(pred_f16)
    );

    // ── Pipeline DUT ──────────────────────────────────────────────────
    logic       busy_pip, done_pip;
    logic [3:0] pred_pip;
    cnn_accel_pipeline u_pip (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy_pip), .done(done_pip), .pred_digit(pred_pip)
    );

    // ── Counters & bookkeeping ────────────────────────────────────────
    int digit, rank, label;
    string fname_i8, fname_i4, fname_f16;

    int total;
    int cor_i8  [0:9], cor_i4  [0:9];
    int cor_f16 [0:9], cor_pip [0:9];
    int tot_d   [0:9];

    // Đo cycle count mỗi ảnh cho từng DUT
    longint cyc_i8_total, cyc_i4_total, cyc_f16_total, cyc_pip_total;
    longint cyc_start;

    initial begin
        // Khởi tạo
        total = 0; cyc_i8_total = 0; cyc_i4_total = 0;
        cyc_f16_total = 0; cyc_pip_total = 0;
        for (int d = 0; d < 10; d++) begin
            cor_i8[d]=0; cor_i4[d]=0; cor_f16[d]=0; cor_pip[d]=0; tot_d[d]=0;
        end

        // Global reset
        rst_n = 0; start = 0;
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5)  @(posedge clk);

        // ── Vòng lặp chính ───────────────────────────────────────────
        for (digit = 0; digit < 10; digit++) begin
            for (rank = 0; rank < 100; rank++) begin
                label = digit;

                // Tên file theo từng thư mục mem/
                if (rank < 10) begin
                    $sformat(fname_i8,  "mem/batch/image_%0d_0%0d.hex", digit, rank);
                    $sformat(fname_i4,  "mem/int4/batch/image_%0d_0%0d.hex", digit, rank);
                    $sformat(fname_f16, "mem/fp16/batch/image_%0d_0%0d.hex", digit, rank);
                end else begin
                    $sformat(fname_i8,  "mem/batch/image_%0d_%0d.hex", digit, rank);
                    $sformat(fname_i4,  "mem/int4/batch/image_%0d_%0d.hex", digit, rank);
                    $sformat(fname_f16, "mem/fp16/batch/image_%0d_%0d.hex", digit, rank);
                end

                // Nạp ảnh vào từng DUT (mỗi DUT dùng quantization riêng)
                $readmemh(fname_i8,  u_int8.image_mem);
                $readmemh(fname_i4,  u_int4.image_mem);
                $readmemh(fname_f16, u_fp16.image_mem);
                $readmemh(fname_i8,  u_pip.image_mem);  // pipeline dùng INT8 mem

                // Reset ngắn
                rst_n = 0;
                repeat(4) @(posedge clk);
                rst_n = 1;
                repeat(2) @(posedge clk);

                // Ghi cycle bắt đầu
                cyc_start = $time / 20; // 20ns per cycle

                // Start tất cả cùng lúc
                start = 1;
                @(posedge clk);
                start = 0;

                // Chờ cả 4 DUT done
                wait(done_i8  == 1); cyc_i8_total  += ($time/20 - cyc_start);
                wait(done_i4  == 1); cyc_i4_total  += ($time/20 - cyc_start);
                wait(done_f16 == 1); cyc_f16_total += ($time/20 - cyc_start);
                wait(done_pip == 1); cyc_pip_total += ($time/20 - cyc_start);

                @(posedge clk);

                // Ghi kết quả
                tot_d[label]++;
                total++;
                if (pred_i8  == label) cor_i8[label]++;
                if (pred_i4  == label) cor_i4[label]++;
                if (pred_f16 == label) cor_f16[label]++;
                if (pred_pip == label) cor_pip[label]++;

                $display("[CMP] image_%0d_%02d label=%0d int8=%0d int4=%0d fp16=%0d pip=%0d",
                    digit, rank, label, pred_i8, pred_i4, pred_f16, pred_pip);

                // Chờ done hạ
                wait(done_i8==0 && done_i4==0 && done_f16==0 && done_pip==0);
                repeat(2) @(posedge clk);
            end
        end

        // ── Báo cáo ──────────────────────────────────────────────────
        $display("");
        $display("================================================================");
        $display("  SO SANH DO CHINH XAC — INT8 / INT4 / FP16 / PIPELINE");
        $display("  Tong so mau: %0d (100 mau x 10 chu so)", total);
        $display("================================================================");
        $display("  Digit | INT8(acc) | INT4(acc) | FP16(acc) | PIP(acc)");
        $display("--------|-----------|-----------|-----------|----------");
        for (int d = 0; d < 10; d++) begin
            $display("    %0d   | %6.2f%%   | %6.2f%%   | %6.2f%%   | %6.2f%%",
                d,
                100.0*cor_i8[d] /tot_d[d],
                100.0*cor_i4[d] /tot_d[d],
                100.0*cor_f16[d]/tot_d[d],
                100.0*cor_pip[d]/tot_d[d]);
        end
        $display("--------|-----------|-----------|-----------|----------");
        $display("  AVG   | %6.2f%%   | %6.2f%%   | %6.2f%%   | %6.2f%%",
            100.0*$countones(cor_i8)/total,
            100.0*$countones(cor_i4)/total,
            100.0*$countones(cor_f16)/total,
            100.0*$countones(cor_pip)/total);

        // Tính tổng đúng thủ công
        begin
            int s8=0, s4=0, sf=0, sp=0;
            for(int d=0;d<10;d++) begin s8+=cor_i8[d]; s4+=cor_i4[d]; sf+=cor_f16[d]; sp+=cor_pip[d]; end
            $display("--------|-----------|-----------|-----------|----------");
            $display("  TOTAL | %3d/%0d   | %3d/%0d   | %3d/%0d   | %3d/%0d",
                s8,total, s4,total, sf,total, sp,total);
            $display("  ACC   | %6.2f%%   | %6.2f%%   | %6.2f%%   | %6.2f%%",
                100.0*s8/total, 100.0*s4/total, 100.0*sf/total, 100.0*sp/total);
        end
        $display("================================================================");

        // Latency so sánh
        $display("");
        $display("  LATENCY TRUNG BINH MOI ANH (cycles):");
        $display("  INT8     : %0d cycles", cyc_i8_total/total);
        $display("  INT4     : %0d cycles", cyc_i4_total/total);
        $display("  FP16     : %0d cycles", cyc_f16_total/total);
        $display("  Pipeline : %0d cycles  (~%0.1f%% cua INT8)",
            cyc_pip_total/total,
            100.0*cyc_pip_total/cyc_i8_total);
        $display("================================================================");

        #100;
        $finish;
    end
endmodule
