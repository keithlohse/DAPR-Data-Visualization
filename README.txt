DataViz_Tutorial/
├── .gitignore                  # Prevents raw data & temp files from pushing to Git
├── _quarto.yml                 # Controls top navbar, site theme, & global settings
├── DataViz_Tutorial.Rproj      # RStudio project file
├── README.md                   # Repository overview & link to official Figshare dataset
│
├── index.qmd                   # Landing Page: Center Info, Intro, & Setup
├── 01-import-qa.qmd             # Section 1: Import, Merge, & Quality Assurance
├── 02-static-plots.qmd          # Section 2: Static Visualizations (ggplot2 & Seaborn)
├── 03-interactive-plots.qmd     # Section 3: Interactive Visualizations (Plotly)
├── 04-animated-plots.qmd        # Section 4: Animated Visualizations (magick)
│
├── docs/                       # TRACKED ON GITHUB (Generated HTML Website)
│   ├── index.html              # Landing page HTML
│   ├── 01-import-qa.html       # Module 1 HTML
│   ├── 02-static-plots.html    # Module 2 HTML
│   ├── 03-interactive-plots.html
│   ├── 04-animated-plots.html
│   └── site_libs/ / *_files/   # JS/CSS dependencies and rendered plot images
│
└── data/                       # LOCAL ONLY (Ignored by Git)
    └── BDS/
        ├── BDSinfo.txt
        ├── trials/             # Individual trial files from Figshare
        └── output/             # Local parquet, csv, rds, & QA logs