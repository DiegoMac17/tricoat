# list.of.packages <- c("tidyverse", "circlize")
# new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
# if(length(new.packages)) install.packages(new.packages)

library("tidyverse")
library("circlize")

# Parse args
args = commandArgs(trailingOnly=TRUE)
overlap_flag = args[1]
if (overlap_flag == 'y'){
    fn_clin = args[2]
    fn_gen = args[3]
    fn_out_clin = args[4]
    fn_out_gen = args[5]

    # Load data
    img_clin <- read.csv(fn_clin,row.names = 1, header= TRUE)
    img_gen <- read.csv(fn_gen,row.names = 1, header= TRUE)

    # Common columns accross modalities
    common_cols <- intersect(names(img_clin),names(img_gen))
    common_rows <- intersect(row.names(img_clin),row.names(img_gen))

    df_1_filtered <- img_clin[common_cols]
    df_1_filtered <- df_1_filtered[rownames(df_1_filtered) %in% common_rows, ]
    df_2_filtered <- img_gen[common_cols]
    df_2_filtered <- df_2_filtered[rownames(df_2_filtered) %in% common_rows, ]

    # Clinical - imaging chord plot
    pdf(paste(fn_out_clin, '.pdf'))
    mat <- data.matrix(df_1_filtered)
    chordDiagram(mat, annotationTrack = "grid", 
        preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(mat))))))
    # we go back to the first track and customize sector labels
    circos.track(track.index = 1, panel.fun = function(x, y) {
        circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index, 
            facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5))
    }, bg.border = NA) # here set bg.border to NA is important
    circos.clear()
    dev.off()

    # Genetics - imaging chord plot
    pdf(paste(fn_out_gen, '.pdf'))
    mat <- data.matrix(df_2_filtered)
    chordDiagram(mat, annotationTrack = "grid", 
        preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(mat))))))
    # we go back to the first track and customize sector labels
    circos.track(track.index = 1, panel.fun = function(x, y) {
        circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index, 
            facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5))
    }, bg.border = NA) # here set bg.border to NA is important
    circos.clear()
    dev.off()


} else {
    fn_in = args[2]
    fn_out = args[3]

    # Load data
    img_mod <- read.csv(fn_in,row.names = 1, header= TRUE)

    # Clinical - imaging chord plot
    pdf(paste(fn_out, '.pdf'))
    mat <- data.matrix(img_mod)
    chordDiagram(mat, annotationTrack = "grid", 
        preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(mat))))))
    # we go back to the first track and customize sector labels
    circos.track(track.index = 1, panel.fun = function(x, y) {
        circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index, 
            facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5))
    }, bg.border = NA) # here set bg.border to NA is important
    circos.clear()
    dev.off()
}






# fn_clin = '../results/attn_scores_img_clin_set_26_top_k_feat_7.csv'
# fn_gen = '../results/attn_scores_img_gen_set_26_top_k_feat_50.csv'
# fn_out_clin = '../results/img_clin'
# fn_out_gen = '../results/img_gen'

