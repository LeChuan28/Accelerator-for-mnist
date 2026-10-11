module cnn_accel_top #(
    parameter int IMAGE_W       = 28,
    parameter int IMAGE_H       = 28,
    parameter int CONV_CH       = 4,
    parameter int K             = 3,
    parameter int CONV_OUT_W    = 26,
    parameter int CONV_OUT_H    = 26,
    parameter int POOL_OUT_W    = 13,
    parameter int POOL_OUT_H    = 13,
    parameter int FC_IN         = 676,
    parameter int FC_OUT        = 10,
    parameter int CONV_SHIFT    = 7,
    parameter int FC_SHIFT      = 6
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       start,
    output logic       busy,
    output logic       done,
    output logic [3:0] pred_digit
);

    localparam int IMAGE_SIZE       = IMAGE_W * IMAGE_H;
    localparam int CONV_W_SIZE      = CONV_CH * K * K;
    localparam int CONV_OUT_SIZE    = CONV_CH * CONV_OUT_W * CONV_OUT_H;
    localparam int POOL_OUT_SIZE    = CONV_CH * POOL_OUT_W * POOL_OUT_H;
    localparam int FC_W_SIZE        = FC_OUT * FC_IN;

    typedef enum logic [3:0] {
        S_IDLE       = 4'd0,
        S_CONV_INIT  = 4'd1,
        S_CONV_MAC   = 4'd2,
        S_CONV_STORE = 4'd3,
        S_POOL_INIT  = 4'd4,
        S_POOL_SCAN  = 4'd5,
        S_POOL_STORE = 4'd6,
        S_FC_INIT    = 4'd7,
        S_FC_MAC     = 4'd8,
        S_FC_STORE   = 4'd9,
        S_ARG_INIT   = 4'd10,
        S_ARG_SCAN   = 4'd11,
        S_DONE       = 4'd12
    } state_t;

    state_t state;

    logic signed [7:0]  image_mem [0:IMAGE_SIZE-1];
    logic signed [7:0]  conv_w_mem[0:CONV_W_SIZE-1];
    logic signed [31:0] conv_b_mem[0:CONV_CH-1];
    logic signed [7:0]  fc_w_mem  [0:FC_W_SIZE-1];
    logic signed [31:0] fc_b_mem  [0:FC_OUT-1];

    logic signed [15:0] conv_feat [0:CONV_OUT_SIZE-1];
    logic signed [15:0] pool_feat [0:POOL_OUT_SIZE-1];
    logic signed [31:0] fc_score  [0:FC_OUT-1];

    logic [1:0]  conv_oc;
    logic [4:0]  conv_x;
    logic [4:0]  conv_y;
    logic [1:0]  conv_kx;
    logic [1:0]  conv_ky;

    logic [1:0]  pool_oc;
    logic [3:0]  pool_x;
    logic [3:0]  pool_y;
    logic        pool_kx;
    logic        pool_ky;

    logic [3:0]  fc_cls;
    logic [9:0]  fc_idx;

    logic [3:0]  arg_idx;
    logic [3:0]  best_cls;
    logic signed [31:0] best_score;

    logic signed [31:0] acc;
    logic signed [15:0] pool_max;

    integer i;

    function automatic int img_idx(input int y, input int x);
        img_idx = y * IMAGE_W + x;
    endfunction

    function automatic int conv_w_idx(input int oc, input int ky, input int kx);
        conv_w_idx = oc * (K * K) + ky * K + kx;
    endfunction

    function automatic int conv_feat_idx(input int oc, input int y, input int x);
        conv_feat_idx = oc * (CONV_OUT_H * CONV_OUT_W) + y * CONV_OUT_W + x;
    endfunction

    function automatic int pool_feat_idx(input int oc, input int y, input int x);
        pool_feat_idx = oc * (POOL_OUT_H * POOL_OUT_W) + y * POOL_OUT_W + x;
    endfunction

    function automatic int fc_w_idx(input int cls, input int idx);
        fc_w_idx = cls * FC_IN + idx;
    endfunction

    function automatic logic signed [15:0] relu_sat16(input logic signed [31:0] value);
        logic signed [31:0] tmp;
        begin
            tmp = value;
            if (tmp < 0)
                relu_sat16 = 16'sd0;
            else if (tmp > 32'sd32767)
                relu_sat16 = 16'sd32767;
            else
                relu_sat16 = tmp[15:0];
        end
    endfunction

    initial begin
        $readmemh("mem/image_0.hex", image_mem);
        $readmemh("mem/conv_w.hex", conv_w_mem);
        $readmemh("mem/conv_b.hex", conv_b_mem);
        $readmemh("mem/fc_w.hex", fc_w_mem);
        $readmemh("mem/fc_b.hex", fc_b_mem);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            busy       <= 1'b0;
            done       <= 1'b0;
            pred_digit <= 4'd0;
            conv_oc    <= '0;
            conv_x     <= '0;
            conv_y     <= '0;
            conv_kx    <= '0;
            conv_ky    <= '0;
            pool_oc    <= '0;
            pool_x     <= '0;
            pool_y     <= '0;
            pool_kx    <= '0;
            pool_ky    <= '0;
            fc_cls     <= '0;
            fc_idx     <= '0;
            arg_idx    <= '0;
            best_cls   <= '0;
            best_score <= -32'sd2147483647;
            acc        <= '0;
            pool_max   <= -16'sd32768;
            for (i = 0; i < FC_OUT; i = i + 1)
                fc_score[i] <= '0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy       <= 1'b1;
                        pred_digit <= 4'd0;
                        conv_oc    <= 2'd0;
                        conv_x     <= 5'd0;
                        conv_y     <= 5'd0;
                        state      <= S_CONV_INIT;
                    end
                end

                S_CONV_INIT: begin
                    acc     <= conv_b_mem[conv_oc];
                    conv_kx <= 2'd0;
                    conv_ky <= 2'd0;
                    state   <= S_CONV_MAC;
                end

                S_CONV_MAC: begin
                    acc <= acc
                        + $signed(image_mem[img_idx(conv_y + conv_ky, conv_x + conv_kx)])
                        * $signed(conv_w_mem[conv_w_idx(conv_oc, conv_ky, conv_kx)]);

                    if (conv_kx == K - 1) begin
                        conv_kx <= 2'd0;
                        if (conv_ky == K - 1) begin
                            state <= S_CONV_STORE;
                        end else begin
                            conv_ky <= conv_ky + 2'd1;
                        end
                    end else begin
                        conv_kx <= conv_kx + 2'd1;
                    end
                end

                S_CONV_STORE: begin
                    conv_feat[conv_feat_idx(conv_oc, conv_y, conv_x)] <= relu_sat16(acc >>> CONV_SHIFT);
                    if (conv_x == CONV_OUT_W - 1) begin
                        conv_x <= 5'd0;
                        if (conv_y == CONV_OUT_H - 1) begin
                            conv_y <= 5'd0;
                            if (conv_oc == CONV_CH - 1) begin
                                pool_oc <= 2'd0;
                                pool_x  <= 4'd0;
                                pool_y  <= 4'd0;
                                state   <= S_POOL_INIT;
                            end else begin
                                conv_oc <= conv_oc + 2'd1;
                                state   <= S_CONV_INIT;
                            end
                        end else begin
                            conv_y <= conv_y + 5'd1;
                            state  <= S_CONV_INIT;
                        end
                    end else begin
                        conv_x <= conv_x + 5'd1;
                        state  <= S_CONV_INIT;
                    end
                end

                S_POOL_INIT: begin
                    pool_kx  <= 1'b0;
                    pool_ky  <= 1'b0;
                    pool_max <= -16'sd32768;
                    state    <= S_POOL_SCAN;
                end

                S_POOL_SCAN: begin
                    logic signed [15:0] current_val;
                    current_val = conv_feat[
                        conv_feat_idx(pool_oc, pool_y * 2 + pool_ky, pool_x * 2 + pool_kx)
                    ];
                    if (current_val > pool_max)
                        pool_max <= current_val;

                    if (pool_kx == 1'b1) begin
                        pool_kx <= 1'b0;
                        if (pool_ky == 1'b1) begin
                            state <= S_POOL_STORE;
                        end else begin
                            pool_ky <= 1'b1;
                        end
                    end else begin
                        pool_kx <= 1'b1;
                    end
                end

                S_POOL_STORE: begin
                    pool_feat[pool_feat_idx(pool_oc, pool_y, pool_x)] <= pool_max;
                    if (pool_x == POOL_OUT_W - 1) begin
                        pool_x <= 4'd0;
                        if (pool_y == POOL_OUT_H - 1) begin
                            pool_y <= 4'd0;
                            if (pool_oc == CONV_CH - 1) begin
                                fc_cls <= 4'd0;
                                state  <= S_FC_INIT;
                            end else begin
                                pool_oc <= pool_oc + 2'd1;
                                state   <= S_POOL_INIT;
                            end
                        end else begin
                            pool_y <= pool_y + 4'd1;
                            state  <= S_POOL_INIT;
                        end
                    end else begin
                        pool_x <= pool_x + 4'd1;
                        state  <= S_POOL_INIT;
                    end
                end

                S_FC_INIT: begin
                    acc    <= fc_b_mem[fc_cls];
                    fc_idx <= 10'd0;
                    state  <= S_FC_MAC;
                end

                S_FC_MAC: begin
                    acc <= acc
                        + $signed(pool_feat[fc_idx])
                        * $signed(fc_w_mem[fc_w_idx(fc_cls, fc_idx)]);
                    if (fc_idx == FC_IN - 1) begin
                        state <= S_FC_STORE;
                    end else begin
                        fc_idx <= fc_idx + 10'd1;
                    end
                end

                S_FC_STORE: begin
                    fc_score[fc_cls] <= acc >>> FC_SHIFT;
                    if (fc_cls == FC_OUT - 1) begin
                        state <= S_ARG_INIT;
                    end else begin
                        fc_cls <= fc_cls + 4'd1;
                        state  <= S_FC_INIT;
                    end
                end

                S_ARG_INIT: begin
                    best_cls   <= 4'd0;
                    best_score <= fc_score[0];
                    arg_idx    <= 4'd1;
                    state      <= S_ARG_SCAN;
                end

                S_ARG_SCAN: begin
                    if (fc_score[arg_idx] > best_score) begin
                        best_score <= fc_score[arg_idx];
                        best_cls   <= arg_idx;
                    end
                    if (arg_idx == FC_OUT - 1) begin
                        pred_digit <= best_cls;
                        state      <= S_DONE;
                    end else begin
                        arg_idx <= arg_idx + 4'd1;
                    end
                end

                S_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
