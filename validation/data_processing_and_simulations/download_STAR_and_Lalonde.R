# Load required packages
# Install and load required packages


select_star_columns <- function(star_data, suffix) {
  base_cols <- c("gender", "ethnicity", "birth")
  all_cols <- colnames(star_data)
  suffix_cols <- all_cols[grepl(paste0(suffix, "$"), all_cols)]
  selected_cols <- c(base_cols, suffix_cols)

  return(star_data[, selected_cols, drop = FALSE])
}

# Load STAR dataset
library(AER)
# install.packages("AER", dependencies = T)
data(STAR)
## reshape data from wide into long format
## 1. variables and their levels
nam <- c("star", "read", "math", "lunch", "school", "degree", "ladder",
         "experience", "tethnicity", "system", "schoolid")
lev <- c("k", "1", "2", "3")
## 2. reshaping
star <- reshape(STAR, idvar = "id", ids = row.names(STAR),
                times = lev, timevar = "grade", direction = "long",
                varying = lapply(nam, function(x) paste(x, lev, sep = "")))
## 3. improve variable names and type
names(star)[5:15] <- nam
star$id <- factor(star$id)
star$grade <- factor(star$grade, levels = lev, labels = c("kindergarten", "1st", "2nd", "3rd"))

na_counts_per_column <- colSums(is.na(star))
print(na_counts_per_column)
star_filtered_data <- star[complete.cases(star), ]

write.csv(star_filtered_data, file = "filtered_STAR_dataset.csv", row.names = FALSE)

# Load Lalonde dataset
# install.packages("designmatch", dependencies = T)
library(designmatch)
data(lalonde)
lalonde_filtered_data <- lalonde[complete.cases(lalonde), ]
write.csv(lalonde_filtered_data, file = "filtered_lalonde_dataset.csv", row.names = FALSE)


# Double ML Data
# install.packages('DoubleML', dependencies = T)
library(DoubleML)
data_401k <- fetch_401k(return_type = "data.table", instrument = TRUE)
write.csv(data_401k, file = "filtered_401k_data.csv", row.names = FALSE)
