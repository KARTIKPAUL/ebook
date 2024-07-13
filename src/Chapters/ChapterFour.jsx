import React from "react";
//var latex = require('react-latex')

function ChapterFour(){
    return(
//             <div className="w-full text-token-text-primary" dir="auto" id="ch-4"
//                                         data-testid="conversation-turn-11" data-scroll-anchor="false">
//                                         <div className="py-2 juice:py-[18px] px-3 text-base md:px-4 m-auto">
//                                             <div
//                                                 className="mx-auto flex flex-1 gap-3 text-base juice:gap-4 juice:md:gap-6 md:max-w-3xl lg:max-w-[40rem] xl:max-w-[48rem]">
                                                
//                                                 <div
//                                                     className="group/conversation-turn relative flex w-full min-w-0 flex-col agent-turn">
//                                                     <div className="flex-col gap-1 md:gap-3">
//                                                         <div className="flex flex-grow flex-col max-w-full">
//                                                             <div data-message-author-role="assistant"
//                                                                 data-message-id="5fcbbfdd-643d-45af-99a1-f83fd32e2608"
//                                                                 dir="auto"
//                                                                 className="text-message flex flex-col items-start whitespace-pre-wrap break-words [.text-message+&amp;]:mt-5 juice:w-full juice:items-end overflow-x-auto gap-2">
//                                                                 <div
//                                                                     className="flex w-full flex-col gap-1 juice:empty:hidden juice:first:pt-[3px]">
//                                                                     <div
//                                                                         className="markdown prose w-full break-words dark:prose-invert dark">
//                                                                         <h3>Chapter 4: Exploratory Data Analysis (EDA)
//                                                                         </h3>
//                                                                         <hr/>
//                                                                         <p>Exploratory Data Analysis (EDA) is a critical
//                                                                             step in the data science process, involving
//                                                                             the use of statistical and graphical
//                                                                             techniques to understand and summarize the
//                                                                             main characteristics of a dataset. EDA helps
//                                                                             uncover patterns, spot anomalies, test
//                                                                             hypotheses, and check assumptions through
//                                                                             visual and quantitative means. This chapter
//                                                                             will guide you through the essential
//                                                                             concepts, tools, and techniques used in EDA
//                                                                             to help you make informed decisions and
//                                                                             prepare your data for modeling.</p>
//                                                                         <h4>Objectives of Exploratory Data Analysis</h4>
//                                                                         <p>The primary goals of EDA include:</p>
//                                                                         <ol>
//                                                                             <li><strong>Understanding the Data</strong>:
//                                                                                 Gaining insights into the dataset's
//                                                                                 structure, patterns, and relationships.
//                                                                             </li>
//                                                                             <li><strong>Identifying Anomalies</strong>:
//                                                                                 Detecting outliers, missing values, and
//                                                                                 inconsistencies in the data.</li>
//                                                                             <li><strong>Generating Hypotheses</strong>:
//                                                                                 Formulating questions and hypotheses
//                                                                                 about the data that can be tested later.
//                                                                             </li>
//                                                                             <li><strong>Informing Modeling
//                                                                                     Choices</strong>: Guiding the
//                                                                                 selection of appropriate machine
//                                                                                 learning algorithms and preprocessing
//                                                                                 steps.</li>
//                                                                         </ol>
//                                                                         <h4>Techniques for Data Visualization</h4>
//                                                                         <p>Data visualization is a powerful tool in EDA
//                                                                             that helps in understanding data
//                                                                             distribution, identifying trends, and
//                                                                             spotting anomalies. Several Python libraries
//                                                                             are commonly used for data visualization,
//                                                                             including Matplotlib, Seaborn, and Pandas
//                                                                             plotting.</p>
//                                                                         <p><strong>1. Matplotlib</strong></p>
//                                                                         <p>Matplotlib is a versatile plotting library in
//                                                                             Python that provides a wide range of 2D
//                                                                             plotting capabilities.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> matplotlib.pyplot <span className="hljs-keyword">as</span> plt

// <span className="hljs-comment"># Line plot</span>
// plt.plot([<span className="hljs-number">1</span>, <span className="hljs-number">2</span>, <span className="hljs-number">3</span>, <span className="hljs-number">4</span>], [<span className="hljs-number">1</span>, <span className="hljs-number">4</span>, <span className="hljs-number">9</span>, <span className="hljs-number">16</span>])
// plt.xlabel(<span className="hljs-string">'x'</span>)
// plt.ylabel(<span className="hljs-string">'y'</span>)
// plt.title(<span className="hljs-string">'Line Plot'</span>)
// plt.show()

// <span className="hljs-comment"># Scatter plot</span>
// plt.scatter([<span className="hljs-number">1</span>, <span className="hljs-number">2</span>, <span className="hljs-number">3</span>, <span className="hljs-number">4</span>], [<span className="hljs-number">1</span>, <span className="hljs-number">4</span>, <span className="hljs-number">9</span>, <span className="hljs-number">16</span>])
// plt.xlabel(<span className="hljs-string">'x'</span>)
// plt.ylabel(<span className="hljs-string">'y'</span>)
// plt.title(<span className="hljs-string">'Scatter Plot'</span>)
// plt.show()

// <span className="hljs-comment"># Bar plot</span>
// plt.bar([<span className="hljs-string">'A'</span>, <span className="hljs-string">'B'</span>, <span className="hljs-string">'C'</span>, <span className="hljs-string">'D'</span>], [<span className="hljs-number">3</span>, <span className="hljs-number">7</span>, <span className="hljs-number">2</span>, <span className="hljs-number">5</span>])
// plt.xlabel(<span className="hljs-string">'Categories'</span>)
// plt.ylabel(<span className="hljs-string">'Values'</span>)
// plt.title(<span className="hljs-string">'Bar Plot'</span>)
// plt.show()
// </code></div></pre>
//                                                                         <p><strong>2. Seaborn</strong></p>
//                                                                         <p>Seaborn is built on top of Matplotlib and
//                                                                             provides a high-level interface for drawing
//                                                                             attractive statistical graphics.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> seaborn <span className="hljs-keyword">as</span> sns

// <span className="hljs-comment"># Load dataset</span>
// data = sns.load_dataset(<span className="hljs-string">"iris"</span>)

// <span className="hljs-comment"># Pair plot</span>
// sns.pairplot(data, hue=<span className="hljs-string">"species"</span>)
// plt.show()

// <span className="hljs-comment"># Distribution plot</span>
// sns.histplot(data[<span className="hljs-string">'sepal_length'</span>], kde=<span className="hljs-literal">True</span>)
// plt.show()

// <span className="hljs-comment"># Box plot</span>
// sns.boxplot(x=<span className="hljs-string">"species"</span>, y=<span className="hljs-string">"sepal_length"</span>, data=data)
// plt.show()
// </code></div></pre>
//                                                                         <p><strong>3. Pandas Plotting</strong></p>
//                                                                         <p>Pandas provides built-in plotting
//                                                                             capabilities that are convenient for quick
//                                                                             and easy data visualization.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> pandas <span className="hljs-keyword">as</span> pd

// <span className="hljs-comment"># Load dataset</span>
// data = pd.read_csv(<span className="hljs-string">'data.csv'</span>)

// <span className="hljs-comment"># Line plot</span>
// data[<span className="hljs-string">'column_name'</span>].plot(kind=<span className="hljs-string">'line'</span>)
// plt.show()

// <span className="hljs-comment"># Scatter plot</span>
// data.plot(kind=<span className="hljs-string">'scatter'</span>, x=<span className="hljs-string">'column1'</span>, y=<span className="hljs-string">'column2'</span>)
// plt.show()

// <span className="hljs-comment"># Histogram</span>
// data[<span className="hljs-string">'column_name'</span>].plot(kind=<span className="hljs-string">'hist'</span>)
// plt.show()
// </code></div></pre>
//                                                                         <h4>Summarizing Data with Descriptive Statistics
//                                                                         </h4>
//                                                                         <p>Descriptive statistics provide a summary of
//                                                                             the data through measures of central
//                                                                             tendency, dispersion, and shape.</p>
//                                                                         <p><strong>1. Central Tendency</strong></p>
//                                                                         <p>Measures of central tendency describe the
//                                                                             center or typical value of a dataset.</p>
         
         
// <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python">

// <span className="hljs-comment"># Mean</span>
// mean_value = data[<span className="hljs-string">'column_name'</span>].mean()
// <span className="hljs-built_in">print</span>(<span className="hljs-string">f`Mean: <span classNameName="hljs-subst">{'{mean_value}'}</span>`</span>)

// <span className="hljs-comment"># Median</span>
// median_value = data[<span className="hljs-string">'column_name'</span>].median()
// <span className="hljs-built_in">print</span>(<span className="hljs-string">f`Median: <span className="hljs-subst">{'{median_value}'}</span>`</span>)

// <span className="hljs-comment"># Mode</span>
// mode_value = data[<span className="hljs-string">'column_name'</span>].mode()[<span className="hljs-number">0</span>]
// <span className="hljs-built_in">print</span>(<span className="hljs-string">f`Mode: <span className="hljs-subst">{'{mode_value}'}</span>`</span>)
// </code></div></pre>
//                                                                         <p><strong>2. Dispersion</strong></p>
//                                                                         <p>Measures of dispersion describe the spread or
//                                                                             variability of the data.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Standard deviation</span>
// std_dev = data[<span className="hljs-string">'column_name'</span>].std()
// <span className="hljs-built_in">print</span>(<span className="hljs-string">f'Standard Deviation: <span className="hljs-subst">{'{std_dev}'}</span>'</span>)

// <span className="hljs-comment"># Variance</span>
// variance = data[<span className="hljs-string">'column_name'</span>].var()
// <span className="hljs-built_in">print</span>(<span className="hljs-string">f'Variance: <span className="hljs-subst">{'{variance}'}</span>'</span>)

// <span className="hljs-comment"># Range</span>
// range_value = data[<span className="hljs-string">'column_name'</span>].<span className="hljs-built_in">max</span>() - data[<span className="hljs-string">'column_name'</span>].<span className="hljs-built_in">min</span>()
// <span className="hljs-built_in">print</span>(<span className="hljs-string">f'Range: <span className="hljs-subst">{'{range_value}'}</span>'</span>)
// </code></div></pre>
//                                                                         <p><strong>3. Shape</strong></p>
//                                                                         <p>Measures of shape describe the distribution's
//                                                                             symmetry and peakedness.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Skewness</span>
// skewness = data[<span className="hljs-string">'column_name'</span>].skew()
// <span className="hljs-built_in">print</span>(<span className="hljs-string">f'Skewness: <span className="hljs-subst">{'{skewness}'}</span>'</span>)

// <span className="hljs-comment"># Kurtosis</span>
// kurtosis = data[<span className="hljs-string">'column_name'</span>].kurt()
// <span className="hljs-built_in">print</span>(<span className="hljs-string">f'Kurtosis: <span className="hljs-subst">{'{kurtosis}'}</span>'</span>)
// </code></div></pre>
//                                                                         <h4>Identifying Patterns and Relationships</h4>
//                                                                         <p>Understanding relationships between variables
//                                                                             is crucial in EDA. Several techniques can
//                                                                             help identify these relationships.</p>
//                                                                         <p><strong>1. Correlation Analysis</strong></p>
//                                                                         <p>Correlation measures the strength and
//                                                                             direction of the relationship between two
//                                                                             variables.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Correlation matrix</span>
// correlation_matrix = data.corr()
// <span className="hljs-built_in">print</span>(correlation_matrix)

// <span className="hljs-comment"># Heatmap of correlation matrix</span>
// sns.heatmap(correlation_matrix, annot=<span className="hljs-literal">True</span>, cmap=<span className="hljs-string">'coolwarm'</span>)
// plt.show()
// </code></div></pre>
//                                                                         <p><strong>2. Cross-Tabulation</strong></p>
//                                                                         <p>Cross-tabulation, or contingency table,
//                                                                             summarizes the relationship between
//                                                                             categorical variables.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Cross-tabulation</span>
// cross_tab = pd.crosstab(data[<span className="hljs-string">'category1'</span>], data[<span className="hljs-string">'category2'</span>])
// <span className="hljs-built_in">print</span>(cross_tab)

// <span className="hljs-comment"># Heatmap of cross-tabulation</span>
// sns.heatmap(cross_tab, annot=<span className="hljs-literal">True</span>, cmap=<span className="hljs-string">'YlGnBu'</span>)
// plt.show()
// </code></div></pre>
//                                                                         <p><strong>3. Grouping and Aggregation</strong>
//                                                                         </p>
//                                                                         <p>Grouping and aggregation help summarize data
//                                                                             by categories.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Group by and aggregate</span>
// grouped_data = data.groupby(<span className="hljs-string">'category_column'</span>).agg({<span className="hljs-string">'numeric_column'</span>} [<span className="hljs-string">'mean'</span>, <span className="hljs-string">'sum'</span>, <span className="hljs-string">'count'</span>])
// <span className="hljs-built_in">print</span>(grouped_data)
// </code></div></pre>
//                                                                         <h4>Handling Outliers and Anomalies</h4>
//                                                                         <p>Outliers and anomalies can skew your analysis
//                                                                             and model performance. Identifying and
//                                                                             handling them is crucial.</p>
//                                                                         <p><strong>1. Visualization Techniques</strong>
//                                                                         </p>
//                                                                         <p>Visualizations like box plots and scatter
//                                                                             plots can help identify outliers.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Box plot to identify outliers</span>
// sns.boxplot(data=data, x=<span className="hljs-string">'numeric_column'</span>)
// plt.show()

// <span className="hljs-comment"># Scatter plot to identify outliers</span>
// sns.scatterplot(data=data, x=<span className="hljs-string">'numeric_column1'</span>, y=<span className="hljs-string">'numeric_column2'</span>)
// plt.show()
// </code></div></pre>
//                                                                         <p><strong>2. Statistical Methods</strong></p>
//                                                                         <p>Statistical methods can quantify outliers.
//                                                                         </p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Z-score method</span>
// <span className="hljs-keyword">from</span> scipy.stats <span className="hljs-keyword">import</span> zscore

// data[<span className="hljs-string">'z_score'</span>] = zscore(data[<span className="hljs-string">'numeric_column'</span>])
// outliers = data[data[<span className="hljs-string">'z_score'</span>].<span className="hljs-built_in">abs</span>() &gt; <span className="hljs-number">3</span>]
// <span className="hljs-built_in">print</span>(outliers)

// <span className="hljs-comment"># IQR method</span>
// Q1 = data[<span className="hljs-string">'numeric_column'</span>].quantile(<span className="hljs-number">0.25</span>)
// Q3 = data[<span className="hljs-string">'numeric_column'</span>].quantile(<span className="hljs-number">0.75</span>)
// IQR = Q3 - Q1
// outliers = data[(data[<span className="hljs-string">'numeric_column'</span>] &lt; (Q1 - <span className="hljs-number">1.5</span> * IQR)) | (data[<span className="hljs-string">'numeric_column'</span>] &gt; (Q3 + <span className="hljs-number">1.5</span> * IQR))]
// <span className="hljs-built_in">print</span>(outliers)
// </code></div></pre>
//                                                                         <p><strong>3. Handling Outliers</strong></p>
//                                                                         <p>Once identified, outliers can be handled by
//                                                                             removing, capping, or transforming them.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Removing outliers</span>
// data_cleaned = data[(data[<span className="hljs-string">'z_score'</span>].<span className="hljs-built_in">abs</span>() &lt;= <span className="hljs-number">3</span>)]

// <span className="hljs-comment"># Capping outliers</span>
// data[<span className="hljs-string">'capped_column'</span>] = data[<span className="hljs-string">'numeric_column'</span>].clip(lower=Q1 - <span className="hljs-number">1.5</span> * IQR, upper=Q3 + <span className="hljs-number">1.5</span> * IQR)

// <span className="hljs-comment"># Transforming outliers</span>
// data[<span className="hljs-string">'log_transformed'</span>] = np.log1p(data[<span className="hljs-string">'numeric_column'</span>])
// </code></div></pre>
//                                                                         <h4>Dealing with Missing Values</h4>
//                                                                         <p>Missing data can significantly impact your
//                                                                             analysis. Several strategies exist to handle
//                                                                             missing values.</p>
//                                                                         <p><strong>1. Visualization of Missing
//                                                                                 Data</strong></p>
//                                                                         <p>Visualizations can help understand the extent
//                                                                             and pattern of missing data.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Heatmap of missing values</span>
// sns.heatmap(data.isnull(), cbar=<span className="hljs-literal">False</span>, cmap=<span className="hljs-string">'viridis'</span>)
// plt.show()

// <span className="hljs-comment"># Bar plot of missing values</span>
// data.isnull().<span className="hljs-built_in">sum</span>().plot(kind=<span className="hljs-string">'bar'</span>)
// plt.show()
// </code></div></pre>
//                                                                         <p><strong>2. Imputation Techniques</strong></p>
//                                                                         <p>Imputation involves filling in missing values
//                                                                             with specific values.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> sklearn.impute <span className="hljs-keyword">import</span> SimpleImputer

// <span className="hljs-comment"># Mean imputation</span>
// mean_imputer = SimpleImputer(strategy=<span className="hljs-string">'mean'</span>)
// data[<span className="hljs-string">'imputed_column'</span>] = mean_imputer.fit_transform(data[[<span className="hljs-string">'column_with_missing'</span>]])

// <span className="hljs-comment"># Median imputation</span>
// median_imputer = SimpleImputer(strategy=<span className="hljs-string">'median'</span>)
// data[<span className="hljs-string">'imputed_column'</span>] = median_imputer.fit_transform(data[[<span className="hljs-string">'column_with_missing'</span>]])

// <span className="hljs-comment"># Mode imputation</span>
// mode_imputer = SimpleImputer(strategy=<span className="hljs-string">'most_frequent'</span>)
// data[<span className="hljs-string">'imputed_column'</span>] = mode_imputer.fit_transform(data[[<span className="hljs-string">'column_with_missing'</span>]])
// </code></div></pre>
//                                                                         <p><strong>3. Advanced Imputation
//                                                                                 Techniques</strong></p>
//                                                                         <p>Advanced techniques consider relationships
//                                                                             between features.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> sklearn.impute <span className="hljs-keyword">import</span> KNNImputer

// <span className="hljs-comment"># K-Nearest Neighbors imputation</span>
// knn_imputer = KNNImputer(n_neighbors=<span className="hljs-number">5</span>)
// data_imputed = knn_imputer.fit_transform(data)
// </code></div></pre>
//                                                                         <h4>Practical EDA Workflow</h4>
//                                                                         <p>Here is a step-by-step workflow for
//                                                                             conducting EDA:</p>
//                                                                         <p><strong>1. Load and Inspect Data</strong></p>
//                                                                         <p>Start by loading your dataset and inspecting
//                                                                             its structure.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Load dataset</span>
// data = pd.read_csv(<span className="hljs-string">'data.csv'</span>)

// <span className="hljs-comment"># Inspect data</span>
// <span className="hljs-built_in">print</span>(data.head())
// <span className="hljs-built_in">print</span>(data.info())
// <span className="hljs-built_in">print</span>(data.describe())
// </code></div></pre>
//                                                                         <p><strong>2. Visualize Data</strong></p>
//                                                                         <p>Use visualizations to understand data
//                                                                             distribution, relationships, and anomalies.
//                                                                         </p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Distribution plots</span>
// sns.histplot(data[<span className="hljs-string">'numeric_column'</span>], kde=<span className="hljs-literal">True</span>)
// plt.show()

// <span className="hljs-comment"># Pair plots</span>
// sns.pairplot(data)
// plt.show()

// <span className="hljs-comment"># Correlation heatmap</span>
// correlation_matrix = data.corr()
// sns.heatmap(correlation_matrix, annot=<span className="hljs-literal">True</span>, cmap=<span className="hljs-string">'coolwarm'</span>)
// plt.show()
// </code></div></pre>
//                                                                         <p><strong>3. Handle Missing Values</strong></p>
//                                                                         <p>Identify and impute or remove missing values.
//                                                                         </p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Visualize missing data</span>
// sns.heatmap(data.isnull(), cbar=<span className="hljs-literal">False</span>, cmap=<span className="hljs-string">'viridis'</span>)
// plt.show()

// <span className="hljs-comment"># Impute missing values</span>
// mean_imputer = SimpleImputer(strategy=<span className="hljs-string">'mean'</span>)
// data[<span className="hljs-string">'imputed_column'</span>] = mean_imputer.fit_transform(data[[<span className="hljs-string">'column_with_missing'</span>]])
// </code></div></pre>
//                                                                         <p><strong>4. Identify and Handle
//                                                                                 Outliers</strong></p>
//                                                                         <p>Detect and handle outliers to ensure data
//                                                                             quality.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Box plot to identify outliers</span>
// sns.boxplot(data=data, x=<span className="hljs-string">'numeric_column'</span>)
// plt.show()

// <span className="hljs-comment"># Remove outliers using Z-score</span>
// data[<span className="hljs-string">'z_score'</span>] = zscore(data[<span className="hljs-string">'numeric_column'</span>])
// data_cleaned = data[(data[<span className="hljs-string">'z_score'</span>].<span className="hljs-built_in">abs</span>() &lt;= <span className="hljs-number">3</span>)]
// </code></div></pre>
//                                                                         <p><strong>5. Summarize and Document
//                                                                                 Findings</strong></p>
//                                                                         <p>Summarize the key findings and document the
//                                                                             EDA process.</p>
//                                                                         <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Summary statistics</span>
// summary = data.describe()
// <span className="hljs-built_in">print</span>(summary)

// <span className="hljs-comment"># Document findings</span>
// <span className="hljs-keyword">with</span> <span className="hljs-built_in">open</span>(<span className="hljs-string">'eda_summary.txt'</span>, <span className="hljs-string">'w'</span>) <span className="hljs-keyword">as</span> file:
//     file.write(<span className="hljs-built_in">str</span>(summary))
// </code></div></pre>
//                                                                         <h4>Conclusion</h4>
//                                                                         <p>Exploratory Data Analysis (EDA) is a vital
//                                                                             step in the data science process that helps
//                                                                             you understand your data, identify patterns
//                                                                             and relationships, and prepare it for
//                                                                             modeling. By employing various visualization
//                                                                             techniques, descriptive statistics, and
//                                                                             methods for handling outliers and missing
//                                                                             values, you can gain valuable insights and
//                                                                             make informed decisions about the subsequent
//                                                                             steps in your machine learning workflow.
//                                                                             This chapter has provided a comprehensive
//                                                                             guide to EDA, equipping you with the tools
//                                                                             and techniques needed to effectively explore
//                                                                             and analyze your data.</p>
//                                                                     </div>
//                                                                 </div>
//                                                             </div>
//                                                         </div>
//                                                     </div>
//                                                 </div>
//                                             </div>
//                                         </div>
//             </div>

<div>
<div className="w-full text-token-text-primary" dir="auto" id="ch-4" data-testid="conversation-turn-11" data-scroll-anchor="false">
  <div className="py-2 juice:py-[18px] px-3 text-base md:px-4 m-auto">
    <div className="mx-auto flex flex-1 gap-3 text-base juice:gap-4 juice:md:gap-6 md:max-w-3xl lg:max-w-[40rem] xl:max-w-[48rem]">
      <div className="group/conversation-turn relative flex w-full min-w-0 flex-col agent-turn">
        <div className="flex-col gap-1 md:gap-3">
          <div className="flex flex-grow flex-col max-w-full">
            <div data-message-author-role="assistant" data-message-id="5fcbbfdd-643d-45af-99a1-f83fd32e2608" dir="auto" className="text-message flex flex-col items-start whitespace-pre-wrap break-words [.text-message+&]:mt-5 juice:w-full juice:items-end overflow-x-auto gap-2">
              <div className="flex w-full flex-col gap-1 juice:empty:hidden juice:first:pt-[3px]">
                <div className="markdown prose w-full break-words dark:prose-invert dark">
                  <h3>Chapter 4: Exploratory Data Analysis (EDA)
                  </h3>
                  <hr />
                  <p>Exploratory Data Analysis (EDA) is a critical
                    step in the data science process, involving
                    the use of statistical and graphical
                    techniques to understand and summarize the
                    main characteristics of a dataset. EDA helps
                    uncover patterns, spot anomalies, test
                    hypotheses, and check assumptions through
                    visual and quantitative means. This chapter
                    will guide you through the essential
                    concepts, tools, and techniques used in EDA
                    to help you make informed decisions and
                    prepare your data for modeling.</p>
                  <h4>Objectives of Exploratory Data Analysis</h4>
                  <p>The primary goals of EDA include:</p>
                  <ol>
                    <li><strong>Understanding the Data</strong>:
                      Gaining insights into the dataset's
                      structure, patterns, and relationships.
                    </li>
                    <li><strong>Identifying Anomalies</strong>:
                      Detecting outliers, missing values, and
                      inconsistencies in the data.</li>
                    <li><strong>Generating Hypotheses</strong>:
                      Formulating questions and hypotheses
                      about the data that can be tested later.
                    </li>
                    <li><strong>Informing Modeling
                        Choices</strong>: Guiding the
                      selection of appropriate machine
                      learning algorithms and preprocessing
                      steps.</li>
                  </ol>
                  <h4>Techniques for Data Visualization</h4>
                  <p>Data visualization is a powerful tool in EDA
                    that helps in understanding data
                    distribution, identifying trends, and
                    spotting anomalies. Several Python libraries
                    are commonly used for data visualization,
                    including Matplotlib, Seaborn, and Pandas
                    plotting.</p>
                  <p><strong>1. Matplotlib</strong></p>
                  <p>Matplotlib is a versatile plotting library in
                    Python that provides a wide range of 2D
                    plotting capabilities.</p>
                  <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> matplotlib.pyplot <span className="hljs-keyword">as</span> plt{"\n"}{"\n"}<span className="hljs-comment"># Line plot</span>{"\n"}plt.plot([<span className="hljs-number">1</span>, <span className="hljs-number">2</span>, <span className="hljs-number">3</span>, <span className="hljs-number">4</span>], [<span className="hljs-number">1</span>, <span className="hljs-number">4</span>, <span className="hljs-number">9</span>, <span className="hljs-number">16</span>]){"\n"}plt.xlabel(<span className="hljs-string">'x'</span>){"\n"}plt.ylabel(<span className="hljs-string">'y'</span>){"\n"}plt.title(<span className="hljs-string">'Line Plot'</span>){"\n"}plt.show(){"\n"}{"\n"}<span className="hljs-comment"># Scatter plot</span>{"\n"}plt.scatter([<span className="hljs-number">1</span>, <span className="hljs-number">2</span>, <span className="hljs-number">3</span>, <span className="hljs-number">4</span>], [<span className="hljs-number">1</span>, <span className="hljs-number">4</span>, <span className="hljs-number">9</span>, <span className="hljs-number">16</span>]){"\n"}plt.xlabel(<span className="hljs-string">'x'</span>){"\n"}plt.ylabel(<span className="hljs-string">'y'</span>){"\n"}plt.title(<span className="hljs-string">'Scatter Plot'</span>){"\n"}plt.show(){"\n"}{"\n"}<span className="hljs-comment"># Bar plot</span>{"\n"}plt.bar([<span className="hljs-string">'A'</span>, <span className="hljs-string">'B'</span>, <span className="hljs-string">'C'</span>, <span className="hljs-string">'D'</span>], [<span className="hljs-number">3</span>, <span className="hljs-number">7</span>, <span className="hljs-number">2</span>, <span className="hljs-number">5</span>]){"\n"}plt.xlabel(<span className="hljs-string">'Categories'</span>){"\n"}plt.ylabel(<span className="hljs-string">'Values'</span>){"\n"}plt.title(<span className="hljs-string">'Bar Plot'</span>){"\n"}plt.show(){"\n"}</code></div></pre></div>
                <p><strong>2. Seaborn</strong></p>
                <p>Seaborn is built on top of Matplotlib and
                  provides a high-level interface for drawing
                  attractive statistical graphics.</p>
                <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> seaborn <span className="hljs-keyword">as</span> sns{"\n"}{"\n"}<span className="hljs-comment"># Load dataset</span>{"\n"}data = sns.load_dataset(<span className="hljs-string">"iris"</span>){"\n"}{"\n"}<span className="hljs-comment"># Pair plot</span>{"\n"}sns.pairplot(data, hue=<span className="hljs-string">"species"</span>){"\n"}plt.show(){"\n"}{"\n"}<span className="hljs-comment"># Distribution plot</span>{"\n"}sns.histplot(data[<span className="hljs-string">'sepal_length'</span>], kde=<span className="hljs-literal">True</span>){"\n"}plt.show(){"\n"}{"\n"}<span className="hljs-comment"># Box plot</span>{"\n"}sns.boxplot(x=<span className="hljs-string">"species"</span>, y=<span className="hljs-string">"sepal_length"</span>, data=data){"\n"}plt.show(){"\n"}</code></div></pre></div>
              <p><strong>3. Pandas Plotting</strong></p>
              <p>Pandas provides built-in plotting
                capabilities that are convenient for quick
                and easy data visualization.</p>
              <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> pandas <span className="hljs-keyword">as</span> pd{"\n"}{"\n"}<span className="hljs-comment"># Load dataset</span>{"\n"}data = pd.read_csv(<span className="hljs-string">'data.csv'</span>){"\n"}{"\n"}<span className="hljs-comment"># Line plot</span>{"\n"}data[<span className="hljs-string">'column_name'</span>].plot(kind=<span className="hljs-string">'line'</span>){"\n"}plt.show(){"\n"}{"\n"}<span className="hljs-comment"># Scatter plot</span>{"\n"}data.plot(kind=<span className="hljs-string">'scatter'</span>, x=<span className="hljs-string">'column1'</span>, y=<span className="hljs-string">'column2'</span>){"\n"}plt.show(){"\n"}{"\n"}<span className="hljs-comment"># Histogram</span>{"\n"}data[<span className="hljs-string">'column_name'</span>].plot(kind=<span className="hljs-string">'hist'</span>){"\n"}plt.show(){"\n"}</code></div></pre></div>
            <h4>Summarizing Data with Descriptive Statistics
            </h4>
            <p>Descriptive statistics provide a summary of
              the data through measures of central
              tendency, dispersion, and shape.</p>
            <p><strong>1. Central Tendency</strong></p>
            <p>Measures of central tendency describe the
              center or typical value of a dataset.</p>
            <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Mean</span>{"\n"}mean_value = data[<span className="hljs-string">'column_name'</span>].mean(){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Mean: <span className="hljs-subst">{"{"}mean_value{"}"}</span>'</span>){"\n"}{"\n"}<span className="hljs-comment"># Median</span>{"\n"}median_value = data[<span className="hljs-string">'column_name'</span>].median(){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Median: <span className="hljs-subst">{"{"}median_value{"}"}</span>'</span>){"\n"}{"\n"}<span className="hljs-comment"># Mode</span>{"\n"}mode_value = data[<span className="hljs-string">'column_name'</span>].mode()[<span className="hljs-number">0</span>]{"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Mode: <span className="hljs-subst">{"{"}mode_value{"}"}</span>'</span>){"\n"}</code></div></pre></div>
          <p><strong>2. Dispersion</strong></p>
          <p>Measures of dispersion describe the spread or
            variability of the data.</p>
          <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Standard deviation</span>{"\n"}std_dev = data[<span className="hljs-string">'column_name'</span>].std(){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Standard Deviation: <span className="hljs-subst">{"{"}std_dev{"}"}</span>'</span>){"\n"}{"\n"}<span className="hljs-comment"># Variance</span>{"\n"}variance = data[<span className="hljs-string">'column_name'</span>].var(){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Variance: <span className="hljs-subst">{"{"}variance{"}"}</span>'</span>){"\n"}{"\n"}<span className="hljs-comment"># Range</span>{"\n"}range_value = data[<span className="hljs-string">'column_name'</span>].<span className="hljs-built_in">max</span>() - data[<span className="hljs-string">'column_name'</span>].<span className="hljs-built_in">min</span>(){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Range: <span className="hljs-subst">{"{"}range_value{"}"}</span>'</span>){"\n"}</code></div></pre></div>
        <p><strong>3. Shape</strong></p>
        <p>Measures of shape describe the distribution's
          symmetry and peakedness.</p>
        <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Skewness</span>{"\n"}skewness = data[<span className="hljs-string">'column_name'</span>].skew(){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Skewness: <span className="hljs-subst">{"{"}skewness{"}"}</span>'</span>){"\n"}{"\n"}<span className="hljs-comment"># Kurtosis</span>{"\n"}kurtosis = data[<span className="hljs-string">'column_name'</span>].kurt(){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Kurtosis: <span className="hljs-subst">{"{"}kurtosis{"}"}</span>'</span>){"\n"}</code></div></pre></div>
      <h4>Identifying Patterns and Relationships</h4>
      <p>Understanding relationships between variables
        is crucial in EDA. Several techniques can
        help identify these relationships.</p>
      <p><strong>1. Correlation Analysis</strong></p>
      <p>Correlation measures the strength and
        direction of the relationship between two
        variables.</p>
      <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Correlation matrix</span>{"\n"}correlation_matrix = data.corr(){"\n"}<span className="hljs-built_in">print</span>(correlation_matrix){"\n"}{"\n"}<span className="hljs-comment"># Heatmap of correlation matrix</span>{"\n"}sns.heatmap(correlation_matrix, annot=<span className="hljs-literal">True</span>, cmap=<span className="hljs-string">'coolwarm'</span>){"\n"}plt.show(){"\n"}</code></div></pre></div>
    <p><strong>2. Cross-Tabulation</strong></p>
    <p>Cross-tabulation, or contingency table,
      summarizes the relationship between
      categorical variables.</p>
    <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Cross-tabulation</span>{"\n"}cross_tab = pd.crosstab(data[<span className="hljs-string">'category1'</span>], data[<span className="hljs-string">'category2'</span>]){"\n"}<span className="hljs-built_in">print</span>(cross_tab){"\n"}{"\n"}<span className="hljs-comment"># Heatmap of cross-tabulation</span>{"\n"}sns.heatmap(cross_tab, annot=<span className="hljs-literal">True</span>, cmap=<span className="hljs-string">'YlGnBu'</span>){"\n"}plt.show(){"\n"}</code></div></pre></div>
  <p><strong>3. Grouping and Aggregation</strong>
  </p>
  <p>Grouping and aggregation help summarize data
    by categories.</p>
  <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Group by and aggregate</span>{"\n"}grouped_data = data.groupby(<span className="hljs-string">'category_column'</span>).agg({"{"}<span className="hljs-string">'numeric_column'</span>: [<span className="hljs-string">'mean'</span>, <span className="hljs-string">'sum'</span>, <span className="hljs-string">'count'</span>]{"}"}){"\n"}<span className="hljs-built_in">print</span>(grouped_data){"\n"}</code></div></pre></div>
<h4>Handling Outliers and Anomalies</h4>
<p>Outliers and anomalies can skew your analysis
  and model performance. Identifying and
  handling them is crucial.</p>
<p><strong>1. Visualization Techniques</strong>
</p>
<p>Visualizations like box plots and scatter
  plots can help identify outliers.</p>
<pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Box plot to identify outliers</span>{"\n"}sns.boxplot(data=data, x=<span className="hljs-string">'numeric_column'</span>){"\n"}plt.show(){"\n"}{"\n"}<span className="hljs-comment"># Scatter plot to identify outliers</span>{"\n"}sns.scatterplot(data=data, x=<span className="hljs-string">'numeric_column1'</span>, y=<span className="hljs-string">'numeric_column2'</span>){"\n"}plt.show(){"\n"}</code></div></pre>
<p><strong>2. Statistical Methods</strong></p>
<p>Statistical methods can quantify outliers.
</p>
<pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Z-score method</span>{"\n"}<span className="hljs-keyword">from</span> scipy.stats <span className="hljs-keyword">import</span> zscore{"\n"}{"\n"}data[<span className="hljs-string">'z_score'</span>] = zscore(data[<span className="hljs-string">'numeric_column'</span>]){"\n"}outliers = data[data[<span className="hljs-string">'z_score'</span>].<span className="hljs-built_in">abs</span>() &gt; <span className="hljs-number">3</span>]{"\n"}<span className="hljs-built_in">print</span>(outliers){"\n"}{"\n"}<span className="hljs-comment"># IQR method</span>{"\n"}Q1 = data[<span className="hljs-string">'numeric_column'</span>].quantile(<span className="hljs-number">0.25</span>){"\n"}Q3 = data[<span className="hljs-string">'numeric_column'</span>].quantile(<span className="hljs-number">0.75</span>){"\n"}IQR = Q3 - Q1{"\n"}outliers = data[(data[<span className="hljs-string">'numeric_column'</span>] &lt; (Q1 - <span className="hljs-number">1.5</span> * IQR)) | (data[<span className="hljs-string">'numeric_column'</span>] &gt; (Q3 + <span className="hljs-number">1.5</span> * IQR))]{"\n"}<span className="hljs-built_in">print</span>(outliers){"\n"}</code></div></pre>
<p><strong>3. Handling Outliers</strong></p>
<p>Once identified, outliers can be handled by
  removing, capping, or transforming them.</p>
<pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Removing outliers</span>{"\n"}data_cleaned = data[(data[<span className="hljs-string">'z_score'</span>].<span className="hljs-built_in">abs</span>() &lt;= <span className="hljs-number">3</span>)]{"\n"}{"\n"}<span className="hljs-comment"># Capping outliers</span>{"\n"}data[<span className="hljs-string">'capped_column'</span>] = data[<span className="hljs-string">'numeric_column'</span>].clip(lower=Q1 - <span className="hljs-number">1.5</span> * IQR, upper=Q3 + <span className="hljs-number">1.5</span> * IQR){"\n"}{"\n"}<span className="hljs-comment"># Transforming outliers</span>{"\n"}data[<span className="hljs-string">'log_transformed'</span>] = np.log1p(data[<span className="hljs-string">'numeric_column'</span>]){"\n"}</code></div></pre>
<h4>Dealing with Missing Values</h4>
<p>Missing data can significantly impact your
  analysis. Several strategies exist to handle
  missing values.</p>
<p><strong>1. Visualization of Missing
    Data</strong></p>
<p>Visualizations can help understand the extent
  and pattern of missing data.</p>
<pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Heatmap of missing values</span>{"\n"}sns.heatmap(data.isnull(), cbar=<span className="hljs-literal">False</span>, cmap=<span className="hljs-string">'viridis'</span>){"\n"}plt.show(){"\n"}{"\n"}<span className="hljs-comment"># Bar plot of missing values</span>{"\n"}data.isnull().<span className="hljs-built_in">sum</span>().plot(kind=<span className="hljs-string">'bar'</span>){"\n"}plt.show(){"\n"}</code></div></pre>
<p><strong>2. Imputation Techniques</strong></p>
<p>Imputation involves filling in missing values
  with specific values.</p>
<pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> sklearn.impute <span className="hljs-keyword">import</span> SimpleImputer{"\n"}{"\n"}<span className="hljs-comment"># Mean imputation</span>{"\n"}mean_imputer = SimpleImputer(strategy=<span className="hljs-string">'mean'</span>){"\n"}data[<span className="hljs-string">'imputed_column'</span>] = mean_imputer.fit_transform(data[[<span className="hljs-string">'column_with_missing'</span>]]){"\n"}{"\n"}<span className="hljs-comment"># Median imputation</span>{"\n"}median_imputer = SimpleImputer(strategy=<span className="hljs-string">'median'</span>){"\n"}data[<span className="hljs-string">'imputed_column'</span>] = median_imputer.fit_transform(data[[<span className="hljs-string">'column_with_missing'</span>]]){"\n"}{"\n"}<span className="hljs-comment"># Mode imputation</span>{"\n"}mode_imputer = SimpleImputer(strategy=<span className="hljs-string">'most_frequent'</span>){"\n"}data[<span className="hljs-string">'imputed_column'</span>] = mode_imputer.fit_transform(data[[<span className="hljs-string">'column_with_missing'</span>]]){"\n"}</code></div></pre>
<p><strong>3. Advanced Imputation
    Techniques</strong></p>
<p>Advanced techniques consider relationships
  between features.</p>
<pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> sklearn.impute <span className="hljs-keyword">import</span> KNNImputer{"\n"}{"\n"}<span className="hljs-comment"># K-Nearest Neighbors imputation</span>{"\n"}knn_imputer = KNNImputer(n_neighbors=<span className="hljs-number">5</span>){"\n"}data_imputed = knn_imputer.fit_transform(data){"\n"}</code></div></pre>
<h4>Practical EDA Workflow</h4>
<p>Here is a step-by-step workflow for
  conducting EDA:</p>
<p><strong>1. Load and Inspect Data</strong></p>
<p>Start by loading your dataset and inspecting
  its structure.</p>
<pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Load dataset</span>{"\n"}data = pd.read_csv(<span className="hljs-string">'data.csv'</span>){"\n"}{"\n"}<span className="hljs-comment"># Inspect data</span>{"\n"}<span className="hljs-built_in">print</span>(data.head()){"\n"}<span className="hljs-built_in">print</span>(data.info()){"\n"}<span className="hljs-built_in">print</span>(data.describe()){"\n"}</code></div></pre>
<p><strong>2. Visualize Data</strong></p>
<p>Use visualizations to understand data
  distribution, relationships, and anomalies.
</p>
<pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Distribution plots</span>{"\n"}sns.histplot(data[<span className="hljs-string">'numeric_column'</span>], kde=<span className="hljs-literal">True</span>){"\n"}plt.show(){"\n"}{"\n"}<span className="hljs-comment"># Pair plots</span>{"\n"}sns.pairplot(data){"\n"}plt.show(){"\n"}{"\n"}<span className="hljs-comment"># Correlation heatmap</span>{"\n"}correlation_matrix = data.corr(){"\n"}sns.heatmap(correlation_matrix, annot=<span className="hljs-literal">True</span>, cmap=<span className="hljs-string">'coolwarm'</span>){"\n"}plt.show(){"\n"}</code></div></pre>
<p><strong>3. Handle Missing Values</strong></p>
<p>Identify and impute or remove missing values.
</p>
<pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Visualize missing data</span>{"\n"}sns.heatmap(data.isnull(), cbar=<span className="hljs-literal">False</span>, cmap=<span className="hljs-string">'viridis'</span>){"\n"}plt.show(){"\n"}{"\n"}<span className="hljs-comment"># Impute missing values</span>{"\n"}mean_imputer = SimpleImputer(strategy=<span className="hljs-string">'mean'</span>){"\n"}data[<span className="hljs-string">'imputed_column'</span>] = mean_imputer.fit_transform(data[[<span className="hljs-string">'column_with_missing'</span>]]){"\n"}</code></div></pre>
<p><strong>4. Identify and Handle
    Outliers</strong></p>
<p>Detect and handle outliers to ensure data
  quality.</p>
<pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Box plot to identify outliers</span>{"\n"}sns.boxplot(data=data, x=<span className="hljs-string">'numeric_column'</span>){"\n"}plt.show(){"\n"}{"\n"}<span className="hljs-comment"># Remove outliers using Z-score</span>{"\n"}data[<span className="hljs-string">'z_score'</span>] = zscore(data[<span className="hljs-string">'numeric_column'</span>]){"\n"}data_cleaned = data[(data[<span className="hljs-string">'z_score'</span>].<span className="hljs-built_in">abs</span>() &lt;= <span className="hljs-number">3</span>)]{"\n"}</code></div></pre>
<p><strong>5. Summarize and Document
    Findings</strong></p>
<p>Summarize the key findings and document the
  EDA process.</p>
<pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Summary statistics</span>{"\n"}summary = data.describe(){"\n"}<span className="hljs-built_in">print</span>(summary){"\n"}{"\n"}<span className="hljs-comment"># Document findings</span>{"\n"}<span className="hljs-keyword">with</span> <span className="hljs-built_in">open</span>(<span className="hljs-string">'eda_summary.txt'</span>, <span className="hljs-string">'w'</span>) <span className="hljs-keyword">as</span> file:{"\n"}{"    "}file.write(<span className="hljs-built_in">str</span>(summary)){"\n"}</code></div></pre>
<h4>Conclusion</h4>
<p>Exploratory Data Analysis (EDA) is a vital
  step in the data science process that helps
  you understand your data, identify patterns
  and relationships, and prepare it for
  modeling. By employing various visualization
  techniques, descriptive statistics, and
  methods for handling outliers and missing
  values, you can gain valuable insights and
  make informed decisions about the subsequent
  steps in your machine learning workflow.
  This chapter has provided a comprehensive
  guide to EDA, equipping you with the tools
  and techniques needed to effectively explore
  and analyze your data.</p>
</div>
        
    )
}

export default ChapterFour;