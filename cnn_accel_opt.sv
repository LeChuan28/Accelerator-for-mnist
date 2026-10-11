module cnn_accel_opt #(
    parameter int IMAGE_W    = 28,
    parameter int IMAGE_H    = 28,
    parameter int CONV_CH    = 4,
    parameter int K          = 3,
    parameter int CONV_OUT_W = 26,
    parameter int CONV_OUT_H = 26,
    parameter int POOL_OUT_W = 13,
    parameter int POOL_OUT_H = 13,
    parameter int FC_IN      = 676,
    parameter int FC_OUT     = 10,
    parameter int CONV_SHIFT = 7,
    parameter int FC_SHIFT   = 6,
    parameter int FEAT_BITS  = 16,    // (3) 16 = khớp gốc; 8 = tiết kiệm 1/2
    parameter int FC_W_BITS  = 8      // (5) 8 = dùng hex hiện có; 4 = cần hex mới
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       start,
    output logic       busy,
    output logic       done,
    output logic [3:0] pred_digit
);
    localparam int CONV_W_SIZE    = CONV_CH * K * K;
    localparam int CONV_CH_SIZE   = CONV_OUT_W * CONV_OUT_H;          // (4) 1 kênh = 676
    localparam int POOL_OUT_SIZE  = CONV_CH * POOL_OUT_W * POOL_OUT_H;
    localparam int FC_W_SIZE      = FC_OUT * FC_IN;
    localparam int CF_AW          = $clog2(CONV_CH_SIZE);
    localparam logic signed [31:0] FEAT_MAX = (32'sd1 <<< (FEAT_BITS-1)) - 32'sd1;

    typedef enum logic [3:0] {
        S_IDLE=4'd0,
        S_CONV_INIT=4'd1, S_CONV_FETCH=4'd2, S_CONV_MAC=4'd3, S_CONV_STORE=4'd4,
        S_POOL_INIT=4'd5, S_POOL_FETCH=4'd6, S_POOL_CMP=4'd7, S_POOL_STORE=4'd8,
        S_FC_INIT=4'd9,   S_FC_FETCH=4'd10, S_FC_MAC=4'd11,   S_FC_STORE=4'd12,
        S_ARG_INIT=4'd13, S_ARG_SCAN=4'd14, S_DONE=4'd15
    } state_t;
    state_t state;

    // -- Bo nho TRONG SO (van nap tu hex) --
    logic signed [7:0]  conv_w_mem[0:CONV_W_SIZE-1]         /* synthesis ramstyle = "logic" */;
    logic signed [31:0] conv_b_mem[0:CONV_CH-1];
    logic signed [FC_W_BITS-1:0] fc_w_mem [0:FC_W_SIZE-1]   /* synthesis ramstyle = "M4K, no_rw_check" */;
    logic signed [31:0] fc_b_mem  [0:FC_OUT-1];

    // -- draw_ram: RAM anh ve tay (MegaWizard), ghi qua JTAG, RTL chi doc --
    logic [9:0]         draw_addr;
    logic signed [7:0]  draw_q;
    draw_ram u_draw_ram (.address(draw_addr), .clock(clk), .data(8'b0), .wren(1'b0), .q(draw_q));

    // -- (3)(4) conv_feat: chi 1 kenh, rong FEAT_BITS --
    logic signed [FEAT_BITS-1:0] cfch      [0:CONV_CH_SIZE-1]  /* synthesis ramstyle = "M4K, no_rw_check" */;
    logic signed [FEAT_BITS-1:0] pool_feat [0:POOL_OUT_SIZE-1] /* synthesis ramstyle = "M4K, no_rw_check" */;
    logic signed [31:0] fc_score [0:FC_OUT-1];

    logic                        cf_we;
    logic [CF_AW-1:0]            cf_waddr, cf_raddr;
    logic signed [FEAT_BITS-1:0] cf_wdata, cf_rdata;
    always_ff @(posedge clk) begin
        if (cf_we) cfch[cf_waddr] <= cf_wdata;
        cf_rdata <= cfch[cf_raddr];
    end

    // -- Bo dem --
    logic [1:0] conv_oc, conv_kx, conv_ky;
    logic [4:0] conv_x, conv_y;
    logic [3:0] pool_x, pool_y, fc_cls, arg_idx, best_cls;
    logic       pool_kx, pool_ky;
    logic [9:0] fc_idx;

    // -- Thanh ghi pipeline --
    logic signed [7:0]           pipe_w_val;
    logic signed [FEAT_BITS-1:0] pipe_feat_val;
    logic signed [FC_W_BITS-1:0] pipe_fcw_val;
    logic                        pipe_valid;

    // -- Tich luy --
    logic signed [31:0] acc, best_score, nxt_best_score;
    logic signed [FEAT_BITS-1:0] pool_max;
    logic [3:0]         nxt_best_cls;

    int tmp_fi, tmp_cw;
    integer i;

    function automatic int conv_w_idx (input int oc, input int ky, input int kx);
        conv_w_idx = oc*(K*K) + ky*K + kx;
    endfunction
    function automatic int cf_idx (input int y, input int x);
        cf_idx = y*CONV_OUT_W + x;
    endfunction
    function automatic int pool_idx (input int oc, input int y, input int x);
        pool_idx = oc*(POOL_OUT_H*POOL_OUT_W) + y*POOL_OUT_W + x;
    endfunction
    function automatic int fc_w_idx (input int cls, input int idx);
        fc_w_idx = cls*FC_IN + idx;
    endfunction
    function automatic logic signed [FEAT_BITS-1:0] sat_feat (input logic signed [31:0] v);
        logic signed [31:0] r;
        if (v < 0)              r = 0;
        else if (v > FEAT_MAX)  r = FEAT_MAX;
        else                    r = v;
        sat_feat = r[FEAT_BITS-1:0];
    endfunction

    initial begin
        // CHI nap TRONG SO tu hex -- khong con nap anh ROM
        $readmemh("mem/conv_w.hex", conv_w_mem);
        $readmemh("mem/conv_b.hex", conv_b_mem);
        $readmemh("mem/fc_w.hex",   fc_w_mem);
        $readmemh("mem/fc_b.hex",   fc_b_mem);
    end

    // -- Dia chi pixel cho draw_ram: TO HOP tu bo dem (canh nhip chuan) --
    int px_idx;
    always_comb px_idx = (conv_y + conv_ky) * IMAGE_W + (conv_x + conv_kx);
    assign draw_addr = px_idx[9:0];

    // -- Dia chi conv_feat (1 kenh) --
    int cf_waddr_full, cf_raddr_full;
    always_comb begin
        cf_we         = (state == S_CONV_STORE);
        cf_waddr_full = cf_idx(conv_y, conv_x);
        cf_waddr      = cf_waddr_full[CF_AW-1:0];
        cf_wdata      = sat_feat(acc >>> CONV_SHIFT);
        cf_raddr_full = cf_idx(pool_y*2 + pool_ky, pool_x*2 + pool_kx);
        cf_raddr      = cf_raddr_full[CF_AW-1:0];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; busy<=0; done<=0; pred_digit<=0;
            conv_oc<=0; conv_x<=0; conv_y<=0; conv_kx<=0; conv_ky<=0;
            pool_x<=0; pool_y<=0; pool_kx<=0; pool_ky<=0;
            fc_cls<=0; fc_idx<=0; arg_idx<=0;
            best_cls<=0; best_score<=32'sh80000001;
            acc<=0; pool_max<=0; pipe_valid<=0;
            pipe_w_val<=0; pipe_feat_val<=0; pipe_fcw_val<=0;
            for (i=0;i<FC_OUT;i++) fc_score[i]<=0;
        end else begin
            done <= 0;
            case (state)
                S_IDLE: begin
                    busy<=0; pipe_valid<=0;
                    if (start) begin
                        busy<=1; pred_digit<=0;
                        conv_oc<=0; conv_x<=0; conv_y<=0;
                        state<=S_CONV_INIT;
                    end
                end

                S_CONV_INIT: begin
                    acc<=conv_b_mem[conv_oc]; conv_kx<=0; conv_ky<=0;
                    pipe_valid<=0; state<=S_CONV_FETCH;
                end
                S_CONV_FETCH: begin
                    tmp_cw     = conv_w_idx(conv_oc, conv_ky, conv_kx);
                    pipe_w_val <= conv_w_mem[tmp_cw];
                    pipe_valid <= 1;
                    state      <= S_CONV_MAC;
                end
                S_CONV_MAC: begin
                    if (pipe_valid)
                        acc <= acc + $signed(draw_q) * $signed(pipe_w_val);
                    if (conv_kx == K-1) begin
                        conv_kx<=0;
                        if (conv_ky==K-1) begin pipe_valid<=0; state<=S_CONV_STORE; end
                        else begin conv_ky<=conv_ky+2'd1; state<=S_CONV_FETCH; end
                    end else begin conv_kx<=conv_kx+2'd1; state<=S_CONV_FETCH; end
                end
                S_CONV_STORE: begin
                    if (conv_x==CONV_OUT_W-1) begin
                        conv_x<=0;
                        if (conv_y==CONV_OUT_H-1) begin
                            conv_y<=0; pool_x<=0; pool_y<=0;
                            state<=S_POOL_INIT;
                        end else begin conv_y<=conv_y+5'd1; state<=S_CONV_INIT; end
                    end else begin conv_x<=conv_x+5'd1; state<=S_CONV_INIT; end
                end

                S_POOL_INIT: begin
                    pool_kx<=0; pool_ky<=0; pool_max<=0; state<=S_POOL_FETCH;
                end
                S_POOL_FETCH: state<=S_POOL_CMP;
                S_POOL_CMP: begin
                    if (cf_rdata > pool_max) pool_max <= cf_rdata;
                    if (pool_kx==1) begin
                        pool_kx<=0;
                        if (pool_ky==1) state<=S_POOL_STORE;
                        else begin pool_ky<=1; state<=S_POOL_FETCH; end
                    end else begin pool_kx<=1; state<=S_POOL_FETCH; end
                end
                S_POOL_STORE: begin
                    tmp_fi = pool_idx(conv_oc, pool_y, pool_x);
                    pool_feat[tmp_fi] <= pool_max;
                    if (pool_x==POOL_OUT_W-1) begin
                        pool_x<=0;
                        if (pool_y==POOL_OUT_H-1) begin
                            pool_y<=0;
                            if (conv_oc==CONV_CH-1) begin fc_cls<=0; state<=S_FC_INIT; end
                            else begin conv_oc<=conv_oc+2'd1; conv_x<=0; conv_y<=0; state<=S_CONV_INIT; end
                        end else begin pool_y<=pool_y+4'd1; state<=S_POOL_INIT; end
                    end else begin pool_x<=pool_x+4'd1; state<=S_POOL_INIT; end
                end

                S_FC_INIT: begin
                    acc<=fc_b_mem[fc_cls]; fc_idx<=0; pipe_valid<=0; state<=S_FC_FETCH;
                end
                S_FC_FETCH: begin
                    tmp_fi        = fc_w_idx(fc_cls, fc_idx);
                    pipe_feat_val <= pool_feat[fc_idx];
                    pipe_fcw_val  <= fc_w_mem[tmp_fi];
                    pipe_valid    <= 1;
                    state         <= S_FC_MAC;
                end
                S_FC_MAC: begin
                    if (pipe_valid)
                        acc <= acc + $signed(pipe_feat_val) * $signed(pipe_fcw_val);
                    if (fc_idx==FC_IN-1) begin pipe_valid<=0; state<=S_FC_STORE; end
                    else begin fc_idx<=fc_idx+10'd1; state<=S_FC_FETCH; end
                end
                S_FC_STORE: begin
                    fc_score[fc_cls] <= acc >>> FC_SHIFT;
                    if (fc_cls==FC_OUT-1) state<=S_ARG_INIT;
                    else begin fc_cls<=fc_cls+4'd1; state<=S_FC_INIT; end
                end

                S_ARG_INIT: begin
                    best_cls<=0; best_score<=fc_score[0]; arg_idx<=1; state<=S_ARG_SCAN;
                end
                S_ARG_SCAN: begin
                    if (fc_score[arg_idx] > best_score) begin
                        nxt_best_score=fc_score[arg_idx]; nxt_best_cls=arg_idx;
                    end else begin
                        nxt_best_score=best_score; nxt_best_cls=best_cls;
                    end
                    best_score<=nxt_best_score; best_cls<=nxt_best_cls;
                    if (arg_idx==FC_OUT-1) begin pred_digit<=nxt_best_cls; state<=S_DONE; end
                    else arg_idx<=arg_idx+4'd1;
                end

                S_DONE: begin busy<=0; done<=1; state<=S_IDLE; end
                default: state<=S_IDLE;
            endcase
        end
    end
endmodule
