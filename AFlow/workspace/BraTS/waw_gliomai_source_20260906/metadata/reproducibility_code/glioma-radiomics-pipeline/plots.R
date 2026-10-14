library(tidyverse)
library(patchwork)
library(Rtsne)

dir.create("figures", recursive = TRUE, showWarnings = FALSE)
options(dplyr.width = Inf)

theme_adjusted <- theme_minimal(base_size = 6) +
    theme(plot.background = element_rect(fill = "white", colour = NA),
          strip.text = element_text(face = "bold", hjust = 0),
          legend.key.height = unit(0, "cm"),
          legend.position = "bottom",
          panel.spacing = unit(2, "lines"),
          legend.title = element_text(face = "bold", size = 5),
          axis.title = element_text(size = 5),
          axis.text = element_text(size = 5, colour = "black"),
          legend.margin = margin(0, 0, 0, 0))

annotation_labels <- c(
    "Enhancing tumour",
    "Necrotic/non-enhancing\ntumor core",
    "FLAIR hyperintense\nabnormality",
    "Post-tumor resection\ncavity")

preop_df <- read_csv("output/features/preop.csv") %>%
    mutate(data_identifier = "Pre-operative")
postop_df <- read_csv("output/features/postop.csv") %>%
    mutate(data_identifier = "Post-operative")

combined_df <- rbind(preop_df, postop_df) %>%
    mutate(
        mask_label = factor(
            mask_label,
            levels = c(1, 2, 3, 4),
            labels = annotation_labels),
        data_identifier = factor(
            data_identifier, 
            levels = c("Pre-operative", "Post-operative")))

na_rows <- combined_df[, grepl("^original_.*", colnames(combined_df))] %>% 
    apply(2, is.na) %>% 
    rowSums
combined_df[
    na_rows > 0, 
    c("identifier", "mask_path", "mask_label", "data_identifier", "label_sum.brain_t2f",
      "error.brain_t1c", "error.brain_t1n", 
      "error.brain_t2w", "error.brain_t2f")
]

combined_df %>% 
    select(identifier, data_identifier, mask_path, error.brain_t1c, error.brain_t1n, error.brain_t2w, error.brain_t2f) %>% 
    group_by(identifier, data_identifier, mask_path) %>% 
    summarise(has_no_errors = all(
        is.na(error.brain_t1c) & 
            is.na(error.brain_t2w) & 
            is.na(error.brain_t2f) &
            is.na(error.brain_t1n))) %>%
    group_by(has_no_errors) %>%
    summarise(N = n())

combined_df <- combined_df %>%
    filter(na_rows == 0)

combined_df %>%
    subset(mask_label != "Post-tumor resection\ncavity") %>%
    group_by(mask_label) %>%
    summarise(
        preop_mean = mean(original_shape_VoxelVolume[data_identifier == "Pre-operative"]),
        postop_mean = mean(original_shape_VoxelVolume[data_identifier == "Post-operative"]),
        p.val = t.test(
            original_shape_VoxelVolume[data_identifier == "Pre-operative"],
            original_shape_VoxelVolume[data_identifier == "Post-operative"]
        )$p.value
    )

set.seed(42)
tsne_viz <- Rtsne(combined_df[, grepl("^original_.*", colnames(combined_df))],
                  seed = 42)
tsne_viz_df <- tsne_viz$Y %>% 
    as.data.frame() %>%
    mutate(
        identifier = combined_df$identifier,
        mask_label = combined_df$mask_label,
        data_identifier = combined_df$data_identifier)

count_plot <- combined_df %>%
    group_by(mask_label, data_identifier) %>%
    summarise(
        N = length(mask_label)
) %>%
    ggplot(aes(x = N, y = mask_label, fill = data_identifier)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8),
             width = 0.8) +
    geom_text(aes(label = N), position = position_dodge(width = 0.8),
              hjust = 1.1, colour = "white", size = 1.5) +
    theme_adjusted +
    xlab("Number of volumes of interest") +
    ylab("Annotation") +
    scale_fill_brewer(palette = "Set1", name = "Data identifier") +
    scale_y_discrete(limits = rev(annotation_labels)) +
    guides(fill = guide_legend(nrow = 1))

volume_plot <- combined_df %>% 
    ggplot(aes(x = original_shape_VoxelVolume, 
               y = mask_label,
               fill = data_identifier,
               group = paste(mask_label, data_identifier))) +
    geom_point(
        position = position_jitterdodge(dodge.width = 0.8, jitter.width = 0.2),
        alpha = 0.5,
        size = 0.25) +
    geom_boxplot(outlier.colour = NA,
                 position = position_dodge(0.8),
                 alpha = 0.5,
                 size = 0.25) +
    theme_adjusted +
    xlab("Volume (sqmm)") +
    ylab("Annotation") + 
    scale_fill_brewer(palette = "Set1", name = "Data identifier") +
    scale_y_discrete(limits = rev(annotation_labels)) +
    scale_x_continuous(trans = "log10", labels = scales::label_comma()) +
    guides(fill = guide_legend(nrow = 1))

axis_length_plot <- combined_df %>% 
    ggplot(aes(x = original_shape_MajorAxisLength, 
               y = original_shape_LeastAxisLength,
               colour = mask_label)) +
    geom_point(size = 0.5, alpha = 0.4, shape = 1) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.5) +
    theme_adjusted +
    facet_wrap(~ data_identifier) +
    xlab("Major axis length (mm)") +
    ylab("Minor axis length (mm)") + 
    scale_colour_brewer(palette = "Set2", name = "Annotation") +
    guides(colour = guide_legend(
        nrow = 2, 
        override.aes = list(size = 1.5, alpha = 1.0)))

tsne_plot <- tsne_viz_df %>% 
    ggplot(aes(x = V1, 
               y = V2,
               colour = mask_label)) +
    geom_point(size = 0.5, alpha = 0.4, shape = 1) +
    theme_adjusted +
    facet_wrap(~ data_identifier) +
    xlab("t-SNE 1") +
    ylab("t-SNE 2") + 
    scale_colour_brewer(palette = "Set2", name = "Annotation") +
    guides(colour = guide_legend(
        nrow = 2, 
        override.aes = list(size = 1.5, alpha = 1.0)))

free(count_plot) +
    free(volume_plot) +
    (axis_length_plot + 
        tsne_plot + 
        plot_layout(guides = "collect", ncol = 1) &
        theme(legend.position = "bottom")) +
    plot_layout(ncol = 1, heights = c(1, 1, 2.0)) +
    plot_annotation(tag_levels = "A")
ggsave("figures/radiomics_plot.png", width = 3, height = 5.8,
       device = "png", dpi = 300)
ggsave("figures/radiomics_plot.pdf", width = 4, height = 5.8)

