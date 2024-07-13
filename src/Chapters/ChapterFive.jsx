import React from "react";

import "katex/dist/katex.min.css";
import "bootstrap/dist/css/bootstrap.min.css";

function ChapterFive() {
  return(
    <div className="w-full text-token-text-primary" dir="auto" id="ch-5" data-testid="conversation-turn-13" data-scroll-anchor="false">
        <div className="py-2 juice:py-[18px] px-3 text-base md:px-4 m-auto">
          <div className="mx-auto flex flex-1 gap-3 text-base juice:gap-4 juice:md:gap-6 md:max-w-3xl lg:max-w-[40rem] xl:max-w-[48rem]">
            <div className="group/conversation-turn relative flex w-full min-w-0 flex-col agent-turn">
              <div className="flex-col gap-1 md:gap-3">
                <div className="flex flex-grow flex-col max-w-full">
                  <div data-message-author-role="assistant" data-message-id="465f84c6-b8c8-4bdd-bb04-0489ad3d0947" dir="auto" className="text-message flex flex-col items-start whitespace-pre-wrap break-words [.text-message+&]:mt-5 juice:w-full juice:items-end overflow-x-auto gap-2">
                    <div className="flex w-full flex-col gap-1 juice:empty:hidden juice:first:pt-[3px]">
                      <div className="markdown prose w-full break-words dark:prose-invert dark">
                        <h3>Chapter 5: Supervised Learning: Regression
                        </h3>
                        <hr />
                        <p>Supervised learning is a type of machine
                          learning where the algorithm is trained on a
                          labeled dataset. This chapter focuses on
                          regression, a supervised learning technique
                          used for predicting continuous outcomes. We
                          will explore the basics of regression, delve
                          into different types of regression models,
                          and demonstrate how to implement these
                          models using Python and popular libraries
                          like Scikit-Learn.</p>
                        <h4>Introduction to Regression</h4>
                        <p>Regression analysis is a statistical method
                          used to examine the relationship between a
                          dependent variable (target) and one or more
                          independent variables (features). The
                          primary goal of regression is to predict the
                          value of the target variable based on the
                          values of the features.</p>
                        <p><strong>Key Concepts:</strong></p>
                        <ul>
                          <li><strong>Dependent Variable
                              (Target)</strong>: The variable we
                            aim to predict.</li>
                          <li><strong>Independent Variables
                              (Features)</strong>: The variables
                            used to make predictions.</li>
                          <li><strong>Regression Line</strong>: A line
                            that best fits the data points,
                            representing the relationship between
                            the dependent and independent variables.
                          </li>
                        </ul>
                        <h4>Types of Regression Models</h4>
                        <p>There are several types of regression models,
                          each suited for different types of data and
                          relationships. We will cover the most
                          commonly used regression models:</p>
                        <ol>
                          <li><strong>Simple Linear
                              Regression</strong></li>
                          <li><strong>Multiple Linear
                              Regression</strong></li>
                          <li><strong>Polynomial Regression</strong>
                          </li>
                          <li><strong>Ridge Regression</strong></li>
                          <li><strong>Lasso Regression</strong></li>
                        </ol>
                        <h4>Simple Linear Regression</h4>
                        <p>Simple linear regression models the
                          relationship between two variables by
                          fitting a linear equation to observed data.
                          It predicts the target variable as a linear
                          combination of a single feature.</p>
                        <p><strong>Equation:</strong>
                          <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                  <semantics>
                                    <mrow>
                                      <mi>y</mi>
                                      <mo>=</mo>
                                      <msub>
                                        <mi>β</mi>
                                        <mn>0</mn>
                                      </msub>
                                      <mo>+</mo>
                                      <msub>
                                        <mi>β</mi>
                                        <mn>1</mn>
                                      </msub>
                                      <mi>x</mi>
                                    </mrow>
                                    <annotation encoding="application/x-tex">
                                      y = \beta_0 +
                                      \beta_1x
                                    </annotation>
                                  </semantics>
                                </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.625em', verticalAlign: '-0.1944em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">0</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">1</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord mathnormal">x</span></span></span></span></span>
                        </p>
                        <p>Where:</p>
                        <ul>
                          <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <semantics>
                                      <mrow>
                                        <mi>y</mi>
                                      </mrow>
                                      <annotation encoding="application/x-tex">
                                        y</annotation>
                                    </semantics>
                                  </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.625em', verticalAlign: '-0.1944em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span></span></span></span></span>
                            is the target variable.</li>
                          <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <semantics>
                                      <mrow>
                                        <msub>
                                          <mi>β</mi>
                                          <mn>0</mn>
                                        </msub>
                                      </mrow>
                                      <annotation encoding="application/x-tex">
                                        \beta_0
                                      </annotation>
                                    </semantics>
                                  </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">0</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                            is the intercept.</li>
                          <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <semantics>
                                      <mrow>
                                        <msub>
                                          <mi>β</mi>
                                          <mn>1</mn>
                                        </msub>
                                      </mrow>
                                      <annotation encoding="application/x-tex">
                                        \beta_1
                                      </annotation>
                                    </semantics>
                                  </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">1</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                            is the slope of the regression line.
                          </li>
                          <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <semantics>
                                      <mrow>
                                        <mi>x</mi>
                                      </mrow>
                                      <annotation encoding="application/x-tex">
                                        x</annotation>
                                    </semantics>
                                  </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.4306em'}} /><span className="mord mathnormal">x</span></span></span></span></span>
                            is the feature.</li>
                        </ul>
                        <p><strong>Implementation in Python:</strong>
                        </p>
                        <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> numpy <span className="hljs-keyword">as</span> np{"\n"}<span className="hljs-keyword">import</span> pandas <span className="hljs-keyword">as</span> pd{"\n"}<span className="hljs-keyword">import</span> matplotlib.pyplot <span className="hljs-keyword">as</span> plt{"\n"}<span className="hljs-keyword">from</span> sklearn.model_selection <span className="hljs-keyword">import</span> train_test_split{"\n"}<span className="hljs-keyword">from</span> sklearn.linear_model <span className="hljs-keyword">import</span> LinearRegression{"\n"}<span className="hljs-keyword">from</span> sklearn.metrics <span className="hljs-keyword">import</span> mean_squared_error, r2_score{"\n"}{"\n"}<span className="hljs-comment"># Load dataset</span>{"\n"}data = pd.read_csv(<span className="hljs-string">'data.csv'</span>){"\n"}{"\n"}<span className="hljs-comment"># Define feature and target variable</span>{"\n"}X = data[[<span className="hljs-string">'feature_column'</span>]].values{"\n"}y = data[<span className="hljs-string">'target_column'</span>].values{"\n"}{"\n"}<span className="hljs-comment"># Split data into training and test sets</span>{"\n"}X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=<span className="hljs-number">0.3</span>, random_state=<span className="hljs-number">42</span>){"\n"}{"\n"}<span className="hljs-comment"># Create a linear regression model</span>{"\n"}model = LinearRegression(){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train){"\n"}{"\n"}<span className="hljs-comment"># Make predictions</span>{"\n"}y_pred = model.predict(X_test){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}mse = mean_squared_error(y_test, y_pred){"\n"}r2 = r2_score(y_test, y_pred){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Mean Squared Error: <span className="hljs-subst">{"{"}mse{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'R^2 Score: <span className="hljs-subst">{"{"}r2{"}"}</span>'</span>){"\n"}{"\n"}<span className="hljs-comment"># Plot the regression line</span>{"\n"}plt.scatter(X_test, y_test, color=<span className="hljs-string">'blue'</span>){"\n"}plt.plot(X_test, y_pred, color=<span className="hljs-string">'red'</span>){"\n"}plt.xlabel(<span className="hljs-string">'Feature'</span>){"\n"}plt.ylabel(<span className="hljs-string">'Target'</span>){"\n"}plt.title(<span className="hljs-string">'Simple Linear Regression'</span>){"\n"}plt.show(){"\n"}</code></div></pre></div>
                      <h4>Multiple Linear Regression</h4>
                      <p>Multiple linear regression extends simple
                        linear regression by modeling the
                        relationship between the target variable and
                        multiple features.</p>
                      <p><strong>Equation:</strong>
                        <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                <semantics>
                                  <mrow>
                                    <mi>y</mi>
                                    <mo>=</mo>
                                    <msub>
                                      <mi>β</mi>
                                      <mn>0</mn>
                                    </msub>
                                    <mo>+</mo>
                                    <msub>
                                      <mi>β</mi>
                                      <mn>1</mn>
                                    </msub>
                                    <msub>
                                      <mi>x</mi>
                                      <mn>1</mn>
                                    </msub>
                                    <mo>+</mo>
                                    <msub>
                                      <mi>β</mi>
                                      <mn>2</mn>
                                    </msub>
                                    <msub>
                                      <mi>x</mi>
                                      <mn>2</mn>
                                    </msub>
                                    <mo>+</mo>
                                    <mo>…</mo>
                                    <mo>+</mo>
                                    <msub>
                                      <mi>β</mi>
                                      <mi>n</mi>
                                    </msub>
                                    <msub>
                                      <mi>x</mi>
                                      <mi>n</mi>
                                    </msub>
                                  </mrow>
                                  <annotation encoding="application/x-tex">
                                    y = \beta_0 +
                                    \beta_1x_1 +
                                    \beta_2x_2 + \ldots
                                    + \beta_nx_n
                                  </annotation>
                                </semantics>
                              </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.625em', verticalAlign: '-0.1944em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">0</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">1</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">1</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.6667em', verticalAlign: '-0.0833em'}} /><span className="minner">…</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">n</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">n</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                      </p>
                      <p>Where:</p>
                      <ul>
                        <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                  <semantics>
                                    <mrow>
                                      <mi>y</mi>
                                    </mrow>
                                    <annotation encoding="application/x-tex">
                                      y</annotation>
                                  </semantics>
                                </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.625em', verticalAlign: '-0.1944em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span></span></span></span></span>
                          is the target variable.</li>
                        <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                  <semantics>
                                    <mrow>
                                      <msub>
                                        <mi>β</mi>
                                        <mn>0</mn>
                                      </msub>
                                    </mrow>
                                    <annotation encoding="application/x-tex">
                                      \beta_0
                                    </annotation>
                                  </semantics>
                                </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">0</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                          is the intercept.</li>
                        <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                  <semantics>
                                    <mrow>
                                      <msub>
                                        <mi>β</mi>
                                        <mn>1</mn>
                                      </msub>
                                      <mo separator="true">
                                        ,</mo>
                                      <msub>
                                        <mi>β</mi>
                                        <mn>2</mn>
                                      </msub>
                                      <mo separator="true">
                                        ,</mo>
                                      <mo>…</mo>
                                      <mo separator="true">
                                        ,</mo>
                                      <msub>
                                        <mi>β</mi>
                                        <mi>n</mi>
                                      </msub>
                                    </mrow>
                                    <annotation encoding="application/x-tex">
                                      \beta_1,
                                      \beta_2, \ldots,
                                      \beta_n
                                    </annotation>
                                  </semantics>
                                </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">1</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="minner">…</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">n</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                          are the coefficients.</li>
                        <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                  <semantics>
                                    <mrow>
                                      <msub>
                                        <mi>x</mi>
                                        <mn>1</mn>
                                      </msub>
                                      <mo separator="true">
                                        ,</mo>
                                      <msub>
                                        <mi>x</mi>
                                        <mn>2</mn>
                                      </msub>
                                      <mo separator="true">
                                        ,</mo>
                                      <mo>…</mo>
                                      <mo separator="true">
                                        ,</mo>
                                      <msub>
                                        <mi>x</mi>
                                        <mi>n</mi>
                                      </msub>
                                    </mrow>
                                    <annotation encoding="application/x-tex">
                                      x_1, x_2,
                                      \ldots, x_n
                                    </annotation>
                                  </semantics>
                                </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.625em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">1</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="minner">…</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">n</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                          are the features.</li>
                      </ul>
                      <p><strong>Implementation in Python:</strong>
                      </p>
                      <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Define feature and target variables</span>{"\n"}X = data[[<span className="hljs-string">'feature1'</span>, <span className="hljs-string">'feature2'</span>, <span className="hljs-string">'feature3'</span>]].values{"\n"}y = data[<span className="hljs-string">'target_column'</span>].values{"\n"}{"\n"}<span className="hljs-comment"># Split data into training and test sets</span>{"\n"}X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=<span className="hljs-number">0.3</span>, random_state=<span className="hljs-number">42</span>){"\n"}{"\n"}<span className="hljs-comment"># Create a linear regression model</span>{"\n"}model = LinearRegression(){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train){"\n"}{"\n"}<span className="hljs-comment"># Make predictions</span>{"\n"}y_pred = model.predict(X_test){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}mse = mean_squared_error(y_test, y_pred){"\n"}r2 = r2_score(y_test, y_pred){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Mean Squared Error: <span className="hljs-subst">{"{"}mse{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'R^2 Score: <span className="hljs-subst">{"{"}r2{"}"}</span>'</span>){"\n"}{"\n"}<span className="hljs-comment"># Plot the coefficients</span>{"\n"}coefficients = pd.DataFrame(model.coef_, [<span className="hljs-string">'feature1'</span>, <span className="hljs-string">'feature2'</span>, <span className="hljs-string">'feature3'</span>], columns=[<span className="hljs-string">'Coefficient'</span>]){"\n"}<span className="hljs-built_in">print</span>(coefficients){"\n"}</code></div></pre></div>
                    <h4>Polynomial Regression</h4>
                    <p>Polynomial regression models the relationship
                      between the target variable and the features
                      as an nth degree polynomial. It captures
                      non-linear relationships by adding
                      polynomial terms to the features.</p>
                    <p><strong>Equation:</strong>
                      <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                              <semantics>
                                <mrow>
                                  <mi>y</mi>
                                  <mo>=</mo>
                                  <msub>
                                    <mi>β</mi>
                                    <mn>0</mn>
                                  </msub>
                                  <mo>+</mo>
                                  <msub>
                                    <mi>β</mi>
                                    <mn>1</mn>
                                  </msub>
                                  <mi>x</mi>
                                  <mo>+</mo>
                                  <msub>
                                    <mi>β</mi>
                                    <mn>2</mn>
                                  </msub>
                                  <msup>
                                    <mi>x</mi>
                                    <mn>2</mn>
                                  </msup>
                                  <mo>+</mo>
                                  <mo>…</mo>
                                  <mo>+</mo>
                                  <msub>
                                    <mi>β</mi>
                                    <mi>n</mi>
                                  </msub>
                                  <msup>
                                    <mi>x</mi>
                                    <mi>n</mi>
                                  </msup>
                                </mrow>
                                <annotation encoding="application/x-tex">
                                  y = \beta_0 +
                                  \beta_1x +
                                  \beta_2x^2 + \ldots
                                  + \beta_nx^n
                                </annotation>
                              </semantics>
                            </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.625em', verticalAlign: '-0.1944em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">0</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">1</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord mathnormal">x</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1.0085em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8141em'}}><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.6667em', verticalAlign: '-0.0833em'}} /><span className="minner">…</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">n</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.6644em'}}><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">n</span></span></span></span></span></span></span></span></span></span></span></span>
                    </p>
                    <p>Where:</p>
                    <ul>
                      <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                <semantics>
                                  <mrow>
                                    <mi>y</mi>
                                  </mrow>
                                  <annotation encoding="application/x-tex">
                                    y</annotation>
                                </semantics>
                              </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.625em', verticalAlign: '-0.1944em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span></span></span></span></span>
                        is the target variable.</li>
                      <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                <semantics>
                                  <mrow>
                                    <msub>
                                      <mi>β</mi>
                                      <mn>0</mn>
                                    </msub>
                                  </mrow>
                                  <annotation encoding="application/x-tex">
                                    \beta_0
                                  </annotation>
                                </semantics>
                              </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">0</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                        is the intercept.</li>
                      <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                <semantics>
                                  <mrow>
                                    <msub>
                                      <mi>β</mi>
                                      <mn>1</mn>
                                    </msub>
                                    <mo separator="true">
                                      ,</mo>
                                    <msub>
                                      <mi>β</mi>
                                      <mn>2</mn>
                                    </msub>
                                    <mo separator="true">
                                      ,</mo>
                                    <mo>…</mo>
                                    <mo separator="true">
                                      ,</mo>
                                    <msub>
                                      <mi>β</mi>
                                      <mi>n</mi>
                                    </msub>
                                  </mrow>
                                  <annotation encoding="application/x-tex">
                                    \beta_1,
                                    \beta_2, \ldots,
                                    \beta_n
                                  </annotation>
                                </semantics>
                              </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">1</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="minner">…</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">n</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                        are the coefficients.</li>
                      <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                <semantics>
                                  <mrow>
                                    <mi>x</mi>
                                    <mo separator="true">
                                      ,</mo>
                                    <msup>
                                      <mi>x</mi>
                                      <mn>2</mn>
                                    </msup>
                                    <mo separator="true">
                                      ,</mo>
                                    <mo>…</mo>
                                    <mo separator="true">
                                      ,</mo>
                                    <msup>
                                      <mi>x</mi>
                                      <mi>n</mi>
                                    </msup>
                                  </mrow>
                                  <annotation encoding="application/x-tex">
                                    x, x^2, \ldots,
                                    x^n</annotation>
                                </semantics>
                              </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1.0085em', verticalAlign: '-0.1944em'}} /><span className="mord mathnormal">x</span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8141em'}}><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="minner">…</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.6644em'}}><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">n</span></span></span></span></span></span></span></span></span></span></span></span>
                        are the polynomial terms of the feature.
                      </li>
                    </ul>
                    <p><strong>Implementation in Python:</strong>
                    </p>
                    <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> sklearn.preprocessing <span className="hljs-keyword">import</span> PolynomialFeatures{"\n"}{"\n"}<span className="hljs-comment"># Define feature and target variables</span>{"\n"}X = data[[<span className="hljs-string">'feature_column'</span>]].values{"\n"}y = data[<span className="hljs-string">'target_column'</span>].values{"\n"}{"\n"}<span className="hljs-comment"># Transform features to polynomial features</span>{"\n"}poly = PolynomialFeatures(degree=<span className="hljs-number">2</span>){"\n"}X_poly = poly.fit_transform(X){"\n"}{"\n"}<span className="hljs-comment"># Split data into training and test sets</span>{"\n"}X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=<span className="hljs-number">0.3</span>, random_state=<span className="hljs-number">42</span>){"\n"}{"\n"}<span className="hljs-comment"># Create a linear regression model</span>{"\n"}model = LinearRegression(){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train){"\n"}{"\n"}<span className="hljs-comment"># Make predictions</span>{"\n"}y_pred = model.predict(X_test){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}mse = mean_squared_error(y_test, y_pred){"\n"}r2 = r2_score(y_test, y_pred){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Mean Squared Error: <span className="hljs-subst">{"{"}mse{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'R^2 Score: <span className="hljs-subst">{"{"}r2{"}"}</span>'</span>){"\n"}{"\n"}<span className="hljs-comment"># Plot the polynomial regression line</span>{"\n"}plt.scatter(X, y, color=<span className="hljs-string">'blue'</span>){"\n"}plt.plot(X, model.predict(poly.fit_transform(X)), color=<span className="hljs-string">'red'</span>){"\n"}plt.xlabel(<span className="hljs-string">'Feature'</span>){"\n"}plt.ylabel(<span className="hljs-string">'Target'</span>){"\n"}plt.title(<span className="hljs-string">'Polynomial Regression'</span>){"\n"}plt.show(){"\n"}</code></div></pre></div>
                  <h4>Ridge Regression</h4>
                  <p>Ridge regression (also known as Tikhonov
                    regularization) adds a penalty term to the
                    least squares method to prevent overfitting.
                    It is a regularization technique that
                    shrinks the regression coefficients by
                    imposing a penalty proportional to their
                    magnitude.</p>
                  <p><strong>Equation:</strong>
                    <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                            <semantics>
                              <mrow>
                                <mtext>
                                  minimize&nbsp;
                                </mtext>
                                <msubsup>
                                  <mo>∑</mo>
                                  <mrow>
                                    <mi>i</mi>
                                    <mo>=</mo>
                                    <mn>1</mn>
                                  </mrow>
                                  <mi>n</mi>
                                </msubsup>
                                <mo stretchy="false">
                                  (</mo>
                                <msub>
                                  <mi>y</mi>
                                  <mi>i</mi>
                                </msub>
                                <mo>−</mo>
                                <msub>
                                  <mi>β</mi>
                                  <mn>0</mn>
                                </msub>
                                <mo>−</mo>
                                <msubsup>
                                  <mo>∑</mo>
                                  <mrow>
                                    <mi>j</mi>
                                    <mo>=</mo>
                                    <mn>1</mn>
                                  </mrow>
                                  <mi>p</mi>
                                </msubsup>
                                <msub>
                                  <mi>β</mi>
                                  <mi>j</mi>
                                </msub>
                                <msub>
                                  <mi>x</mi>
                                  <mrow>
                                    <mi>i</mi>
                                    <mi>j</mi>
                                  </mrow>
                                </msub>
                                <msup>
                                  <mo stretchy="false">
                                    )</mo>
                                  <mn>2</mn>
                                </msup>
                                <mo>+</mo>
                                <mi>λ</mi>
                                <msubsup>
                                  <mo>∑</mo>
                                  <mrow>
                                    <mi>j</mi>
                                    <mo>=</mo>
                                    <mn>1</mn>
                                  </mrow>
                                  <mi>p</mi>
                                </msubsup>
                                <msubsup>
                                  <mi>β</mi>
                                  <mi>j</mi>
                                  <mn>2</mn>
                                </msubsup>
                              </mrow>
                              <annotation encoding="application/x-tex">
                                \text{'{'}minimize{'}'} \
                                \sum_{'{'}i=1{'}'}^{'{'}n{'}'} (y_i
                                - \beta_0 -
                                \sum_{'{'}j=1{'}'}^{'{'}p{'}'}
                                \beta_j x_{'{'}ij{'}'})^2 +
                                \lambda
                                \sum_{'{'}j=1{'}'}^{'{'}p{'}'}
                                \beta_j^2
                              </annotation>
                            </semantics>
                          </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1.104em', verticalAlign: '-0.2997em'}} /><span className="mord text"><span className="mord">minimize</span></span><span className="mspace">&nbsp;</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mop"><span className="mop op-symbol small-op" style={{position: 'relative', top: '0em'}}>∑</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8043em'}}><span style={{top: '-2.4003em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">i</span><span className="mrel mtight">=</span><span className="mord mtight">1</span></span></span></span><span style={{top: '-3.2029em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">n</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2997em'}}><span /></span></span></span></span></span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '-0.0359em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">i</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">−</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">0</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">−</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1.2499em', verticalAlign: '-0.4358em'}} /><span className="mop"><span className="mop op-symbol small-op" style={{position: 'relative', top: '0em'}}>∑</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8043em'}}><span style={{top: '-2.4003em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05724em'}}>j</span><span className="mrel mtight">=</span><span className="mord mtight">1</span></span></span></span><span style={{top: '-3.2029em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">p</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4358em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05724em'}}>j</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2861em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05724em'}}>ij</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2861em'}}><span /></span></span></span></span></span><span className="mclose"><span className="mclose">)</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8141em'}}><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1.2499em', verticalAlign: '-0.4358em'}} /><span className="mord mathnormal">λ</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mop"><span className="mop op-symbol small-op" style={{position: 'relative', top: '0em'}}>∑</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8043em'}}><span style={{top: '-2.4003em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05724em'}}>j</span><span className="mrel mtight">=</span><span className="mord mtight">1</span></span></span></span><span style={{top: '-3.2029em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">p</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4358em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8141em'}}><span style={{top: '-2.4413em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05724em'}}>j</span></span></span><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.3948em'}}><span /></span></span></span></span></span></span></span></span></span>
                  </p>
                  <p>Where:</p>
                  <ul>
                    <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                              <semantics>
                                <mrow>
                                  <mi>λ</mi>
                                </mrow>
                                <annotation encoding="application/x-tex">
                                  \lambda
                                </annotation>
                              </semantics>
                            </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.6944em'}} /><span className="mord mathnormal">λ</span></span></span></span></span>
                      is the regularization parameter.</li>
                    <li>Other terms are as defined previously.
                    </li>
                  </ul>
                  <p><strong>Implementation in Python:</strong>
                  </p>
                  <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> sklearn.linear_model <span className="hljs-keyword">import</span> Ridge{"\n"}{"\n"}<span className="hljs-comment"># Define feature and target variables</span>{"\n"}X = data[[<span className="hljs-string">'feature1'</span>, <span className="hljs-string">'feature2'</span>, <span className="hljs-string">'feature3'</span>]].values{"\n"}y = data[<span className="hljs-string">'target_column'</span>].values{"\n"}{"\n"}<span className="hljs-comment"># Split data into training and test sets</span>{"\n"}X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=<span className="hljs-number">0.3</span>, random_state=<span className="hljs-number">42</span>){"\n"}{"\n"}<span className="hljs-comment"># Create a ridge regression model</span>{"\n"}model = Ridge(alpha=<span className="hljs-number">1.0</span>){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train){"\n"}{"\n"}<span className="hljs-comment"># Make predictions</span>{"\n"}y_pred = model.predict(X_test){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}mse = mean_squared_error(y_test, y_pred){"\n"}r2 = r2_score(y_test, y_pred){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Mean Squared Error: <span className="hljs-subst">{"{"}mse{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'R^2 Score: <span className="hljs-subst">{"{"}r2{"}"}</span>'</span>){"\n"}</code></div></pre></div>
                <h4>Lasso Regression</h4>
                <p>Lasso regression (Least Absolute Shrinkage
                  and Selection Operator) is another
                  regularization technique that adds a penalty
                  term to the least squares method. It
                  performs both variable selection and
                  regularization, which can result in sparse
                  models with fewer features.</p>
                <p><strong>Equation:</strong>
                  <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                          <semantics>
                            <mrow>
                              <mtext>
                                minimize&nbsp;
                              </mtext>
                              <msubsup>
                                <mo>∑</mo>
                                <mrow>
                                  <mi>i</mi>
                                  <mo>=</mo>
                                  <mn>1</mn>
                                </mrow>
                                <mi>n</mi>
                              </msubsup>
                              <mo stretchy="false">
                                (</mo>
                              <msub>
                                <mi>y</mi>
                                <mi>i</mi>
                              </msub>
                              <mo>−</mo>
                              <msub>
                                <mi>β</mi>
                                <mn>0</mn>
                              </msub>
                              <mo>−</mo>
                              <msubsup>
                                <mo>∑</mo>
                                <mrow>
                                  <mi>j</mi>
                                  <mo>=</mo>
                                  <mn>1</mn>
                                </mrow>
                                <mi>p</mi>
                              </msubsup>
                              <msub>
                                <mi>β</mi>
                                <mi>j</mi>
                              </msub>
                              <msub>
                                <mi>x</mi>
                                <mrow>
                                  <mi>i</mi>
                                  <mi>j</mi>
                                </mrow>
                              </msub>
                              <msup>
                                <mo stretchy="false">
                                  )</mo>
                                <mn>2</mn>
                              </msup>
                              <mo>+</mo>
                              <mi>λ</mi>
                              <msubsup>
                                <mo>∑</mo>
                                <mrow>
                                  <mi>j</mi>
                                  <mo>=</mo>
                                  <mn>1</mn>
                                </mrow>
                                <mi>p</mi>
                              </msubsup>
                              <mi mathvariant="normal">
                                ∣</mi>
                              <msub>
                                <mi>β</mi>
                                <mi>j</mi>
                              </msub>
                              <mi mathvariant="normal">
                                ∣</mi>
                            </mrow>
                            <annotation encoding="application/x-tex">
                              \text{'{'}minimize{'}'} \
                              \sum_{'{'}i=1{'}'}^{'{'}n{'}'} (y_i
                              - \beta_0 -
                              \sum_{'{'}j=1{'}'}^{'{'}p{'}'}
                              \beta_j x_{'{'}ij{'}'})^2 +
                              \lambda
                              \sum_{'{'}j=1{'}'}^{'{'}p{'}'}
                              |\beta_j|
                            </annotation>
                          </semantics>
                        </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1.104em', verticalAlign: '-0.2997em'}} /><span className="mord text"><span className="mord">minimize</span></span><span className="mspace">&nbsp;</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mop"><span className="mop op-symbol small-op" style={{position: 'relative', top: '0em'}}>∑</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8043em'}}><span style={{top: '-2.4003em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">i</span><span className="mrel mtight">=</span><span className="mord mtight">1</span></span></span></span><span style={{top: '-3.2029em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">n</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2997em'}}><span /></span></span></span></span></span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '-0.0359em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">i</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">−</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">0</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">−</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1.2499em', verticalAlign: '-0.4358em'}} /><span className="mop"><span className="mop op-symbol small-op" style={{position: 'relative', top: '0em'}}>∑</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8043em'}}><span style={{top: '-2.4003em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05724em'}}>j</span><span className="mrel mtight">=</span><span className="mord mtight">1</span></span></span></span><span style={{top: '-3.2029em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">p</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4358em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05724em'}}>j</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2861em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05724em'}}>ij</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2861em'}}><span /></span></span></span></span></span><span className="mclose"><span className="mclose">)</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8141em'}}><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1.2401em', verticalAlign: '-0.4358em'}} /><span className="mord mathnormal">λ</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mop"><span className="mop op-symbol small-op" style={{position: 'relative', top: '0em'}}>∑</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8043em'}}><span style={{top: '-2.4003em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05724em'}}>j</span><span className="mrel mtight">=</span><span className="mord mtight">1</span></span></span></span><span style={{top: '-3.2029em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">p</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4358em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord">∣</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05724em'}}>j</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2861em'}}><span /></span></span></span></span></span><span className="mord">∣</span></span></span></span></span>
                </p>
                <p>Where:</p>
                <ul>
                  <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                            <semantics>
                              <mrow>
                                <mi>λ</mi>
                              </mrow>
                              <annotation encoding="application/x-tex">
                                \lambda
                              </annotation>
                            </semantics>
                          </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.6944em'}} /><span className="mord mathnormal">λ</span></span></span></span></span>
                    is the regularization parameter.</li>
                  <li>Other terms are as defined previously.
                  </li>
                </ul>
                <p><strong>Implementation in Python:</strong>
                </p>
                <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> sklearn.linear_model <span className="hljs-keyword">import</span> Lasso{"\n"}{"\n"}<span className="hljs-comment"># Define feature and target variables</span>{"\n"}X = data[[<span className="hljs-string">'feature1'</span>, <span className="hljs-string">'feature2'</span>, <span className="hljs-string">'feature3'</span>]].values{"\n"}y = data[<span className="hljs-string">'target_column'</span>].values{"\n"}{"\n"}<span className="hljs-comment"># Split data into training and test sets</span>{"\n"}X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=<span className="hljs-number">0.3</span>, random_state=<span className="hljs-number">42</span>){"\n"}{"\n"}<span className="hljs-comment"># Create a lasso regression model</span>{"\n"}model = Lasso(alpha=<span className="hljs-number">1.0</span>){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train){"\n"}{"\n"}<span className="hljs-comment"># Make predictions</span>{"\n"}y_pred = model.predict(X_test){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}mse = mean_squared_error(y_test, y_pred){"\n"}r2 = r2_score(y_test, y_pred){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Mean Squared Error: <span className="hljs-subst">{"{"}mse{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'R^2 Score: <span className="hljs-subst">{"{"}r2{"}"}</span>'</span>){"\n"}{"\n"}<span className="hljs-comment"># Plot the coefficients</span>{"\n"}coefficients = pd.DataFrame(model.coef_, [<span className="hljs-string">'feature1'</span>, <span className="hljs-string">'feature2'</span>, <span className="hljs-string">'feature3'</span>], columns=[<span className="hljs-string">'Coefficient'</span>]){"\n"}<span className="hljs-built_in">print</span>(coefficients){"\n"}</code></div></pre></div>
              <h4>Model Evaluation Metrics</h4>
              <p>Evaluating the performance of regression
                models is crucial to understand how well
                they generalize to new data. Common
                evaluation metrics for regression models
                include:</p>
              <ol>
                <li>
                  <p><strong>Mean Squared Error
                      (MSE)</strong>: Measures the
                    average of the squares of the
                    errors. It is sensitive to outliers.
                    <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                            <semantics>
                              <mrow>
                                <mtext>MSE
                                </mtext>
                                <mo>=</mo>
                                <mfrac>
                                  <mn>1
                                  </mn>
                                  <mi>n
                                  </mi>
                                </mfrac>
                                <msubsup>
                                  <mo>∑
                                  </mo>
                                  <mrow>
                                    <mi>i
                                    </mi>
                                    <mo>=
                                    </mo>
                                    <mn>1
                                    </mn>
                                  </mrow>
                                  <mi>n
                                  </mi>
                                </msubsup>
                                <mo stretchy="false">
                                  (</mo>
                                <msub>
                                  <mi>y
                                  </mi>
                                  <mi>i
                                  </mi>
                                </msub>
                                <mo>−</mo>
                                <msub>
                                  <mover accent="true">
                                    <mi>y
                                    </mi>
                                    <mo>^
                                    </mo>
                                  </mover>
                                  <mi>i
                                  </mi>
                                </msub>
                                <msup>
                                  <mo stretchy="false">
                                    )
                                  </mo>
                                  <mn>2
                                  </mn>
                                </msup>
                              </mrow>
                              <annotation encoding="application/x-tex">
                                \text{'{'}MSE{'}'} =
                                \frac{'{'}1{'}'}{'{'}n{'}'}
                                \sum_{'{'}i=1{'}'}^{'{'}n{'}'}
                                (y_i -
                                \hat{'{'}y{'}'}_i)^2
                              </annotation>
                            </semantics>
                          </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.6833em'}} /><span className="mord text"><span className="mord">MSE</span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.1901em', verticalAlign: '-0.345em'}} /><span className="mord"><span className="mopen nulldelimiter" /><span className="mfrac"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8451em'}}><span style={{top: '-2.655em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">n</span></span></span></span><span style={{top: '-3.23em'}}><span className="pstrut" style={{height: '3em'}} /><span className="frac-line" style={{borderBottomWidth: '0.04em'}} /></span><span style={{top: '-3.394em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.345em'}}><span /></span></span></span></span><span className="mclose nulldelimiter" /></span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mop"><span className="mop op-symbol small-op" style={{position: 'relative', top: '0em'}}>∑</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8043em'}}><span style={{top: '-2.4003em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">i</span><span className="mrel mtight">=</span><span className="mord mtight">1</span></span></span></span><span style={{top: '-3.2029em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">n</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2997em'}}><span /></span></span></span></span></span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '-0.0359em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">i</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">−</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1.0641em', verticalAlign: '-0.25em'}} /><span className="mord"><span className="mord accent"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.6944em'}}><span style={{top: '-3em'}}><span className="pstrut" style={{height: '3em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span></span><span style={{top: '-3em'}}><span className="pstrut" style={{height: '3em'}} /><span className="accent-body" style={{left: '-0.1944em'}}><span className="mord">^</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.1944em'}}><span /></span></span></span></span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '-0.0359em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">i</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose"><span className="mclose">)</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8141em'}}><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span></span></span></span></span></span></span></span></span>
                  </p>
                </li>
                <li>
                  <p><strong>Root Mean Squared Error
                      (RMSE)</strong>: The square root
                    of MSE, providing a measure of the
                    average magnitude of the errors.
                    <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                            <semantics>
                              <mrow>
                                <mtext>RMSE
                                </mtext>
                                <mo>=</mo>
                                <msqrt>
                                  <mtext>
                                    MSE
                                  </mtext>
                                </msqrt>
                              </mrow>
                              <annotation encoding="application/x-tex">
                                \text{'{'}RMSE{'}'}
                                =
                                \sqrt{'{'}\text{'{'}MSE{'}'}{'}'}
                              </annotation>
                            </semantics>
                          </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.6833em'}} /><span className="mord text"><span className="mord">RMSE</span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.04em', verticalAlign: '-0.1133em'}} /><span className="mord sqrt"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.9267em'}}><span className="svg-align" style={{top: '-3em'}}><span className="pstrut" style={{height: '3em'}} /><span className="mord" style={{paddingLeft: '0.833em'}}><span className="mord text"><span className="mord">MSE</span></span></span></span><span style={{top: '-2.8867em'}}><span className="pstrut" style={{height: '3em'}} /><span className="hide-tail" style={{minWidth: '0.853em', height: '1.08em'}}><svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice">
                                          <path d="M95,702
c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
c69,-144,104.5,-217.7,106.5,-221
l0 -0
c5.3,-9.3,12,-14,20,-14
H400000v40H845.2724
s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
M834 80h400000v40h-400000z" />
                                        </svg></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.1133em'}}><span /></span></span></span></span></span></span></span></span>
                  </p>
                </li>
                <li>
                  <p><strong>Mean Absolute Error
                      (MAE)</strong>: Measures the
                    average of the absolute errors,
                    providing a measure of the average
                    magnitude of the errors without
                    considering their direction.
                    <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                            <semantics>
                              <mrow>
                                <mtext>MAE
                                </mtext>
                                <mo>=</mo>
                                <mfrac>
                                  <mn>1
                                  </mn>
                                  <mi>n
                                  </mi>
                                </mfrac>
                                <msubsup>
                                  <mo>∑
                                  </mo>
                                  <mrow>
                                    <mi>i
                                    </mi>
                                    <mo>=
                                    </mo>
                                    <mn>1
                                    </mn>
                                  </mrow>
                                  <mi>n
                                  </mi>
                                </msubsup>
                                <mi mathvariant="normal">
                                  ∣</mi>
                                <msub>
                                  <mi>y
                                  </mi>
                                  <mi>i
                                  </mi>
                                </msub>
                                <mo>−</mo>
                                <msub>
                                  <mover accent="true">
                                    <mi>y
                                    </mi>
                                    <mo>^
                                    </mo>
                                  </mover>
                                  <mi>i
                                  </mi>
                                </msub>
                                <mi mathvariant="normal">
                                  ∣</mi>
                              </mrow>
                              <annotation encoding="application/x-tex">
                                \text{'{'}MAE{'}'} =
                                \frac{'{'}1{'}'}{'{'}n{'}'}
                                \sum_{'{'}i=1{'}'}^{'{'}n{'}'}
                                |y_i -
                                \hat{'{'}y{'}'}_i|
                              </annotation>
                            </semantics>
                          </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.6833em'}} /><span className="mord text"><span className="mord">MAE</span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.1901em', verticalAlign: '-0.345em'}} /><span className="mord"><span className="mopen nulldelimiter" /><span className="mfrac"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8451em'}}><span style={{top: '-2.655em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">n</span></span></span></span><span style={{top: '-3.23em'}}><span className="pstrut" style={{height: '3em'}} /><span className="frac-line" style={{borderBottomWidth: '0.04em'}} /></span><span style={{top: '-3.394em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.345em'}}><span /></span></span></span></span><span className="mclose nulldelimiter" /></span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mop"><span className="mop op-symbol small-op" style={{position: 'relative', top: '0em'}}>∑</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8043em'}}><span style={{top: '-2.4003em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">i</span><span className="mrel mtight">=</span><span className="mord mtight">1</span></span></span></span><span style={{top: '-3.2029em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">n</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2997em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord">∣</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '-0.0359em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">i</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">−</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord"><span className="mord accent"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.6944em'}}><span style={{top: '-3em'}}><span className="pstrut" style={{height: '3em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span></span><span style={{top: '-3em'}}><span className="pstrut" style={{height: '3em'}} /><span className="accent-body" style={{left: '-0.1944em'}}><span className="mord">^</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.1944em'}}><span /></span></span></span></span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '-0.0359em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">i</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord">∣</span></span></span></span></span>
                  </p>
                </li>
                <li>
                  <p><strong>R-squared (R²)</strong>:
                    Measures the proportion of the
                    variance in the dependent variable
                    that is predictable from the
                    independent variables.
                    <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                            <semantics>
                              <mrow>
                                <msup>
                                  <mi>R
                                  </mi>
                                  <mn>2
                                  </mn>
                                </msup>
                                <mo>=</mo>
                                <mn>1</mn>
                                <mo>−</mo>
                                <mfrac>
                                  <mrow>
                                    <msubsup>
                                      <mo>∑
                                      </mo>
                                      <mrow>
                                        <mi>i
                                        </mi>
                                        <mo>=
                                        </mo>
                                        <mn>1
                                        </mn>
                                      </mrow>
                                      <mi>n
                                      </mi>
                                    </msubsup>
                                    <mo stretchy="false">
                                      (
                                    </mo>
                                    <msub>
                                      <mi>y
                                      </mi>
                                      <mi>i
                                      </mi>
                                    </msub>
                                    <mo>−
                                    </mo>
                                    <msub>
                                      <mover accent="true">
                                        <mi>y
                                        </mi>
                                        <mo>^
                                        </mo>
                                      </mover>
                                      <mi>i
                                      </mi>
                                    </msub>
                                    <msup>
                                      <mo stretchy="false">
                                        )
                                      </mo>
                                      <mn>2
                                      </mn>
                                    </msup>
                                  </mrow>
                                  <mrow>
                                    <msubsup>
                                      <mo>∑
                                      </mo>
                                      <mrow>
                                        <mi>i
                                        </mi>
                                        <mo>=
                                        </mo>
                                        <mn>1
                                        </mn>
                                      </mrow>
                                      <mi>n
                                      </mi>
                                    </msubsup>
                                    <mo stretchy="false">
                                      (
                                    </mo>
                                    <msub>
                                      <mi>y
                                      </mi>
                                      <mi>i
                                      </mi>
                                    </msub>
                                    <mo>−
                                    </mo>
                                    <mover accent="true">
                                      <mi>y
                                      </mi>
                                      <mo>ˉ
                                      </mo>
                                    </mover>
                                    <msup>
                                      <mo stretchy="false">
                                        )
                                      </mo>
                                      <mn>2
                                      </mn>
                                    </msup>
                                  </mrow>
                                </mfrac>
                              </mrow>
                              <annotation encoding="application/x-tex">
                                R^2 = 1 -
                                \frac{'{'}\sum_{'{'}i=1{'}'}^{'{'}n{'}'}
                                (y_i -
                                \hat{'{'}y{'}'}_i)^2{'}'}{'{'}\sum_{'{'}i=1{'}'}^{'{'}n{'}'}
                                (y_i -
                                \bar{'{'}y{'}'})^2{'}'}
                              </annotation>
                            </semantics>
                          </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8141em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.00773em'}}>R</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8141em'}}><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '0.7278em', verticalAlign: '-0.0833em'}} /><span className="mord">1</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">−</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1.7289em', verticalAlign: '-0.57em'}} /><span className="mord"><span className="mopen nulldelimiter" /><span className="mfrac"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '1.1589em'}}><span style={{top: '-2.655em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mop mtight"><span className="mop op-symbol small-op mtight" style={{position: 'relative', top: '0em'}}>∑</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.7047em'}}><span style={{top: '-2.1786em', marginLeft: '0em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">i</span><span className="mrel mtight">=</span><span className="mord mtight">1</span></span></span></span><span style={{top: '-2.8971em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">n</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.3214em'}}><span /></span></span></span></span></span><span className="mopen mtight">(</span><span className="mord mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.03588em'}}>y</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3281em'}}><span style={{top: '-2.357em', marginLeft: '-0.0359em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mathnormal mtight">i</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.143em'}}><span /></span></span></span></span></span><span className="mbin mtight">−</span><span className="mord accent mtight"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.5678em'}}><span style={{top: '-2.7em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="mord mathnormal mtight" style={{marginRight: '0.03588em'}}>y</span></span><span style={{top: '-2.7em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="accent-body" style={{left: '-0.1944em'}}><span className="mord mtight">ˉ</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.1944em'}}><span /></span></span></span></span><span className="mclose mtight"><span className="mclose mtight">)</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.7463em'}}><span style={{top: '-2.786em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight">2</span></span></span></span></span></span></span></span></span></span></span><span style={{top: '-3.23em'}}><span className="pstrut" style={{height: '3em'}} /><span className="frac-line" style={{borderBottomWidth: '0.04em'}} /></span><span style={{top: '-3.535em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mop mtight"><span className="mop op-symbol small-op mtight" style={{position: 'relative', top: '0em'}}>∑</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.7385em'}}><span style={{top: '-2.1786em', marginLeft: '0em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">i</span><span className="mrel mtight">=</span><span className="mord mtight">1</span></span></span></span><span style={{top: '-2.931em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">n</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.3214em'}}><span /></span></span></span></span></span><span className="mopen mtight">(</span><span className="mord mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.03588em'}}>y</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3281em'}}><span style={{top: '-2.357em', marginLeft: '-0.0359em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mathnormal mtight">i</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.143em'}}><span /></span></span></span></span></span><span className="mbin mtight">−</span><span className="mord mtight"><span className="mord accent mtight"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.6944em'}}><span style={{top: '-2.7em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="mord mathnormal mtight" style={{marginRight: '0.03588em'}}>y</span></span><span style={{top: '-2.7em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="accent-body" style={{left: '-0.1944em'}}><span className="mord mtight">^</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.1944em'}}><span /></span></span></span></span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3281em'}}><span style={{top: '-2.357em', marginLeft: '-0.0359em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mathnormal mtight">i</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.143em'}}><span /></span></span></span></span></span><span className="mclose mtight"><span className="mclose mtight">)</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8913em'}}><span style={{top: '-2.931em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight">2</span></span></span></span></span></span></span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.57em'}}><span /></span></span></span></span><span className="mclose nulldelimiter" /></span></span></span></span></span>
                  </p>
                </li>
              </ol>
              <p><strong>Implementation in Python:</strong>
              </p>
              <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Evaluate the model</span>{"\n"}mse = mean_squared_error(y_test, y_pred){"\n"}rmse = np.sqrt(mse){"\n"}mae = mean_absolute_error(y_test, y_pred){"\n"}r2 = r2_score(y_test, y_pred){"\n"}{"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Mean Squared Error: <span className="hljs-subst">{"{"}mse{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Root Mean Squared Error: <span className="hljs-subst">{"{"}rmse{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Mean Absolute Error: <span className="hljs-subst">{"{"}mae{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'R^2 Score: <span className="hljs-subst">{"{"}r2{"}"}</span>'</span>){"\n"}</code></div></pre></div>
            <h4>Practical Tips for Regression Analysis</h4>
            <p>Here are some practical tips to improve your
              regression models:</p>
            <ol>
              <li><strong>Feature Engineering</strong>:
                Create new features or transform
                existing ones to capture non-linear
                relationships and interactions.</li>
              <li><strong>Feature Selection</strong>: Use
                techniques like backward elimination,
                forward selection, or regularization
                methods to select relevant features.
              </li>
              <li><strong>Model Validation</strong>: Use
                cross-validation to assess the model's
                performance and prevent overfitting.
              </li>
              <li><strong>Residual Analysis</strong>:
                Analyze the residuals (errors) to check
                for patterns, homoscedasticity, and
                normality.</li>
              <li><strong>Hyperparameter Tuning</strong>:
                Optimize the model's hyperparameters
                using grid search or random search to
                improve performance.</li>
            </ol>
            <h4>Conclusion</h4>
            <p>Regression is a fundamental technique in
              supervised learning for predicting
              continuous outcomes. This chapter covered
              the basics of regression, including simple
              linear regression, multiple linear
              regression, polynomial regression, ridge
              regression, and lasso regression. We also
              discussed model evaluation metrics and
              provided practical tips for improving
              regression models. By understanding and
              implementing these regression techniques,
              you can build robust models that accurately
              predict continuous outcomes and make
              informed decisions based on your data.</p>
          </div>
        </div>
      </div>
  )
}

export default ChapterFive;
