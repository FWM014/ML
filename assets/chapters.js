/* Single source of truth for the book's structure.
   Every chapter page loads this and app.js builds the sidebar, pager, and progress from it. */
window.BOOK = {
  title: "Machine Learning: Zero to Expert",
  parts: [
    {
      title: "Part I · Foundations",
      level: "beginner",
      chapters: [
        { num: "01", file: "01-what-is-ml.html",          title: "What Machine Learning Really Is" },
        { num: "02", file: "02-data.html",                title: "Data: The Raw Material" },
        { num: "03", file: "03-math.html",                title: "The Math You Actually Need" },
        { num: "04", file: "04-python-stack.html",        title: "The Python Toolkit" },
        { num: "05", file: "05-linear-regression.html",   title: "Linear Regression & Gradient Descent" },
        { num: "06", file: "06-classification.html",      title: "Classification & Its Metrics" },
        { num: "07", file: "07-generalization.html",      title: "Overfitting, Bias–Variance & Validation" }
      ]
    },
    {
      title: "Part II · The Core Toolkit",
      level: "intermediate",
      chapters: [
        { num: "08", file: "08-feature-engineering.html", title: "Feature Engineering & Pipelines" },
        { num: "09", file: "09-trees.html",               title: "Decision Trees" },
        { num: "10", file: "10-ensembles.html",           title: "Ensembles: Forests & Boosting" },
        { num: "11", file: "11-other-classics.html",      title: "kNN, SVM & Naive Bayes" },
        { num: "12", file: "12-unsupervised.html",        title: "Clustering & Dimensionality Reduction" },
        { num: "13", file: "13-model-selection.html",     title: "Tuning, Imbalance & Model Selection" },
        { num: "14", file: "14-time-series.html",         title: "Time Series Forecasting" }
      ]
    },
    {
      title: "Part III · Deep Learning",
      level: "advanced",
      chapters: [
        { num: "15", file: "15-neural-networks.html",     title: "Neural Networks From Scratch" },
        { num: "16", file: "16-training-deep-nets.html",  title: "Training Deep Networks (PyTorch)" },
        { num: "17", file: "17-cnn.html",                 title: "Convolutional Networks & Vision" },
        { num: "18", file: "18-sequences-transformers.html", title: "Sequences, Attention & Transformers" },
        { num: "19", file: "19-nlp-llms.html",            title: "NLP, Embeddings & Large Language Models" }
      ]
    },
    {
      title: "Part IV · Expert & Production",
      level: "expert",
      chapters: [
        { num: "20", file: "20-interpretability.html",    title: "Explaining Models: SHAP & Friends" },
        { num: "21", file: "21-mlops.html",               title: "MLOps: Shipping & Monitoring Models" },
        { num: "22", file: "22-advanced-topics.html",     title: "Recommenders, Anomalies, Causality & RL" },
        { num: "23", file: "23-capstone.html",            title: "The End-to-End Project Playbook" },
        { num: "24", file: "24-cheatsheets.html",         title: "Cheat Sheets, Glossary & Interview Prep" }
      ]
    }
  ]
};
