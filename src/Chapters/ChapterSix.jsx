import React from "react";
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

function ChapterSix(){
   return(
    <div className="w-full text-token-text-primary" dir="auto" id="ch-6" data-testid="conversation-turn-15" data-scroll-anchor="false">
        <div className="py-2 juice:py-[18px] px-3 text-base md:px-4 m-auto">
          <div className="mx-auto flex flex-1 gap-3 text-base juice:gap-4 juice:md:gap-6 md:max-w-3xl lg:max-w-[40rem] xl:max-w-[48rem]">
            <div className="group/conversation-turn relative flex w-full min-w-0 flex-col agent-turn">
              <div className="flex-col gap-1 md:gap-3">
                <div className="flex flex-grow flex-col max-w-full">
                  <div data-message-author-role="assistant" data-message-id="015c10ee-0584-43dc-b6d1-c05d61a8cb55" dir="auto" className="text-message flex flex-col items-start whitespace-pre-wrap break-words [.text-message+&]:mt-5 juice:w-full juice:items-end overflow-x-auto gap-2">
                    <div className="flex w-full flex-col gap-1 juice:empty:hidden juice:first:pt-[3px]">
                      <div className="markdown prose w-full break-words dark:prose-invert dark">
                        <h3>Chapter 6: Supervised Learning:
                          Classification</h3>
                        <hr />
                        <p>Classification is a type of supervised
                          learning used to predict categorical
                          outcomes. In this chapter, we will explore
                          various classification algorithms,
                          understand their underlying principles, and
                          learn how to implement them using Python and
                          popular libraries like Scikit-Learn. We will
                          cover logistic regression, k-nearest
                          neighbors (k-NN), support vector machines
                          (SVM), decision trees, and random forests.
                          Additionally, we will discuss model
                          evaluation metrics specific to
                          classification tasks.</p>
                        <h4>Introduction to Classification</h4>
                        <p>Classification involves assigning a label to
                          an input based on its features. Unlike
                          regression, which predicts continuous
                          values, classification predicts discrete
                          categories. Examples include spam detection,
                          disease diagnosis, and image recognition.
                        </p>
                        <p><strong>Key Concepts:</strong></p>
                        <ul>
                          <li><strong>Dependent Variable
                              (Target)</strong>: The categorical
                            variable we aim to predict.</li>
                          <li><strong>Independent Variables
                              (Features)</strong>: The variables
                            used to make predictions.</li>
                          <li><strong>Classes</strong>: The possible
                            categories or labels that the target
                            variable can take.</li>
                        </ul>
                        <h4>Logistic Regression</h4>
                        <p>Logistic regression is a linear model for
                          binary classification. It models the
                          probability that an instance belongs to a
                          particular class.</p>
                        <p><strong>Equation:</strong>
                          <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                  <semantics>
                                    <mrow>
                                      <mi>P</mi>
                                      <mo stretchy="false">
                                        (</mo>
                                      <mi>y</mi>
                                      <mo>=</mo>
                                      <mn>1</mn>
                                      <mi mathvariant="normal">
                                        ∣</mi>
                                      <mi>x</mi>
                                      <mo stretchy="false">
                                        )</mo>
                                      <mo>=</mo>
                                      <mfrac>
                                        <mn>1</mn>
                                        <mrow>
                                          <mn>1</mn>
                                          <mo>+</mo>
                                          <msup>
                                            <mi>e
                                            </mi>
                                            <mrow>
                                              <mo>−
                                              </mo>
                                              <mo stretchy="false">
                                                (
                                              </mo>
                                              <msub>
                                                <mi>β
                                                </mi>
                                                <mn>0
                                                </mn>
                                              </msub>
                                              <mo>+
                                              </mo>
                                              <msub>
                                                <mi>β
                                                </mi>
                                                <mn>1
                                                </mn>
                                              </msub>
                                              <msub>
                                                <mi>x
                                                </mi>
                                                <mn>1
                                                </mn>
                                              </msub>
                                              <mo>+
                                              </mo>
                                              <msub>
                                                <mi>β
                                                </mi>
                                                <mn>2
                                                </mn>
                                              </msub>
                                              <msub>
                                                <mi>x
                                                </mi>
                                                <mn>2
                                                </mn>
                                              </msub>
                                              <mo>+
                                              </mo>
                                              <mo>…
                                              </mo>
                                              <mo>+
                                              </mo>
                                              <msub>
                                                <mi>β
                                                </mi>
                                                <mi>n
                                                </mi>
                                              </msub>
                                              <msub>
                                                <mi>x
                                                </mi>
                                                <mi>n
                                                </mi>
                                              </msub>
                                              <mo stretchy="false">
                                                )
                                              </mo>
                                            </mrow>
                                          </msup>
                                        </mrow>
                                      </mfrac>
                                    </mrow>
                                    <annotation encoding="application/x-tex">
                                      P(y=1|x) =
                                      \frac{'{'}1{'}'}{'{'}1 +
                                      e^{'{'}-(\beta_0 +
                                      \beta_1x_1 +
                                      \beta_2x_2 + \ldots
                                      + \beta_nx_n){'}'}{'}'}
                                    </annotation>
                                  </semantics>
                                </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>P</span><span className="mopen">(</span><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord">1∣</span><span className="mord mathnormal">x</span><span className="mclose">)</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.3137em', verticalAlign: '-0.4686em'}} /><span className="mord"><span className="mopen nulldelimiter" /><span className="mfrac"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8451em'}}><span style={{top: '-2.5898em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight">1</span><span className="mbin mtight">+</span><span className="mord mtight"><span className="mord mathnormal mtight">e</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8575em'}}><span style={{top: '-2.8575em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5357em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mtight">−</span><span className="mopen mtight">(</span><span className="mord mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3448em'}}><span style={{top: '-2.3448em', marginLeft: '-0.0528em', marginRight: '0.1em'}}><span className="pstrut" style={{height: '2.6444em'}} /><span className="mord mtight">0</span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2996em'}}><span /></span></span></span></span></span><span className="mbin mtight">+</span><span className="mord mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3448em'}}><span style={{top: '-2.3448em', marginLeft: '-0.0528em', marginRight: '0.1em'}}><span className="pstrut" style={{height: '2.6444em'}} /><span className="mord mtight">1</span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2996em'}}><span /></span></span></span></span></span><span className="mord mtight"><span className="mord mathnormal mtight">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3448em'}}><span style={{top: '-2.3448em', marginLeft: '0em', marginRight: '0.1em'}}><span className="pstrut" style={{height: '2.6444em'}} /><span className="mord mtight">1</span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2996em'}}><span /></span></span></span></span></span><span className="mbin mtight">+</span><span className="mord mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3448em'}}><span style={{top: '-2.3448em', marginLeft: '-0.0528em', marginRight: '0.1em'}}><span className="pstrut" style={{height: '2.6444em'}} /><span className="mord mtight">2</span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2996em'}}><span /></span></span></span></span></span><span className="mord mtight"><span className="mord mathnormal mtight">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3448em'}}><span style={{top: '-2.3448em', marginLeft: '0em', marginRight: '0.1em'}}><span className="pstrut" style={{height: '2.6444em'}} /><span className="mord mtight">2</span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2996em'}}><span /></span></span></span></span></span><span className="mbin mtight">+</span><span className="minner mtight">…</span><span className="mbin mtight">+</span><span className="mord mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2306em'}}><span style={{top: '-2.3em', marginLeft: '-0.0528em', marginRight: '0.1em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="mord mathnormal mtight">n</span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2em'}}><span /></span></span></span></span></span><span className="mord mtight"><span className="mord mathnormal mtight">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2306em'}}><span style={{top: '-2.3em', marginLeft: '0em', marginRight: '0.1em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="mord mathnormal mtight">n</span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2em'}}><span /></span></span></span></span></span><span className="mclose mtight">)</span></span></span></span></span></span></span></span></span></span></span></span><span style={{top: '-3.23em'}}><span className="pstrut" style={{height: '3em'}} /><span className="frac-line" style={{borderBottomWidth: '0.04em'}} /></span><span style={{top: '-3.394em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4686em'}}><span /></span></span></span></span><span className="mclose nulldelimiter" /></span></span></span></span></span>
                        </p>
                        <p>Where:</p>
                        <ul>
                          <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <semantics>
                                      <mrow>
                                        <mi>P</mi>
                                        <mo stretchy="false">
                                          (</mo>
                                        <mi>y</mi>
                                        <mo>=</mo>
                                        <mn>1</mn>
                                        <mi mathvariant="normal">
                                          ∣</mi>
                                        <mi>x</mi>
                                        <mo stretchy="false">
                                          )</mo>
                                      </mrow>
                                      <annotation encoding="application/x-tex">
                                        P(y=1|x)
                                      </annotation>
                                    </semantics>
                                  </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>P</span><span className="mopen">(</span><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord">1∣</span><span className="mord mathnormal">x</span><span className="mclose">)</span></span></span></span></span>
                            is the probability of the instance
                            belonging to class 1.</li>
                          <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <semantics>
                                      <mrow>
                                        <msub>
                                          <mi>β</mi>
                                          <mn>0</mn>
                                        </msub>
                                        <mo separator="true">
                                          ,</mo>
                                        <msub>
                                          <mi>β</mi>
                                          <mn>1</mn>
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
                                        \beta_0,
                                        \beta_1, \ldots,
                                        \beta_n
                                      </annotation>
                                    </semantics>
                                  </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">0</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">1</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="minner">…</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">n</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
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
                        <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> pandas <span className="hljs-keyword">as</span> pd{"\n"}<span className="hljs-keyword">from</span> sklearn.model_selection <span className="hljs-keyword">import</span> train_test_split{"\n"}<span className="hljs-keyword">from</span> sklearn.linear_model <span className="hljs-keyword">import</span> LogisticRegression{"\n"}<span className="hljs-keyword">from</span> sklearn.metrics <span className="hljs-keyword">import</span> accuracy_score, confusion_matrix, classification_report{"\n"}{"\n"}<span className="hljs-comment"># Load dataset</span>{"\n"}data = pd.read_csv(<span className="hljs-string">'data.csv'</span>){"\n"}{"\n"}<span className="hljs-comment"># Define feature and target variables</span>{"\n"}X = data[[<span className="hljs-string">'feature1'</span>, <span className="hljs-string">'feature2'</span>, <span className="hljs-string">'feature3'</span>]].values{"\n"}y = data[<span className="hljs-string">'target_column'</span>].values{"\n"}{"\n"}<span className="hljs-comment"># Split data into training and test sets</span>{"\n"}X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=<span className="hljs-number">0.3</span>, random_state=<span className="hljs-number">42</span>){"\n"}{"\n"}<span className="hljs-comment"># Create a logistic regression model</span>{"\n"}model = LogisticRegression(){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train){"\n"}{"\n"}<span className="hljs-comment"># Make predictions</span>{"\n"}y_pred = model.predict(X_test){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}accuracy = accuracy_score(y_test, y_pred){"\n"}conf_matrix = confusion_matrix(y_test, y_pred){"\n"}class_report = classification_report(y_test, y_pred){"\n"}{"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Accuracy: <span className="hljs-subst">{"{"}accuracy{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">'Confusion Matrix:'</span>){"\n"}<span className="hljs-built_in">print</span>(conf_matrix){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">'Classification Report:'</span>){"\n"}<span className="hljs-built_in">print</span>(class_report){"\n"}</code></div></pre></div>
                      <h4>k-Nearest Neighbors (k-NN)</h4>
                      <p>k-NN is a non-parametric, instance-based
                        learning algorithm. It classifies an
                        instance based on the majority class among
                        its k-nearest neighbors.</p>
                      <p><strong>Algorithm:</strong></p>
                      <ol>
                        <li>Choose the number of neighbors <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                  <semantics>
                                    <mrow>
                                      <mi>k</mi>
                                    </mrow>
                                    <annotation encoding="application/x-tex">
                                      k</annotation>
                                  </semantics>
                                </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.6944em'}} /><span className="mord mathnormal" style={{marginRight: '0.03148em'}}>k</span></span></span></span></span>.
                        </li>
                        <li>Calculate the distance between the
                          instance and all other instances.</li>
                        <li>Select the k-nearest neighbors.</li>
                        <li>Assign the class based on the majority
                          class among the k-nearest neighbors.
                        </li>
                      </ol>
                      <p><strong>Implementation in Python:</strong>
                      </p>
                      <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> sklearn.neighbors <span className="hljs-keyword">import</span> KNeighborsClassifier{"\n"}{"\n"}<span className="hljs-comment"># Create a k-NN model</span>{"\n"}model = KNeighborsClassifier(n_neighbors=<span className="hljs-number">5</span>){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train){"\n"}{"\n"}<span className="hljs-comment"># Make predictions</span>{"\n"}y_pred = model.predict(X_test){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}accuracy = accuracy_score(y_test, y_pred){"\n"}conf_matrix = confusion_matrix(y_test, y_pred){"\n"}class_report = classification_report(y_test, y_pred){"\n"}{"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Accuracy: <span className="hljs-subst">{"{"}accuracy{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">'Confusion Matrix:'</span>){"\n"}<span className="hljs-built_in">print</span>(conf_matrix){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">'Classification Report:'</span>){"\n"}<span className="hljs-built_in">print</span>(class_report){"\n"}</code></div></pre></div>
                    <h4>Support Vector Machines (SVM)</h4>
                    <p>SVM is a powerful classification algorithm
                      that finds the hyperplane that best
                      separates the classes in the feature space.
                      It can handle linear and non-linear
                      classification using kernel functions.</p>
                    <p><strong>Equation:</strong></p>
                    <p>For linear SVM:
                      <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                              <semantics>
                                <mrow>
                                  <mi>f</mi>
                                  <mo stretchy="false">
                                    (</mo>
                                  <mi>x</mi>
                                  <mo stretchy="false">
                                    )</mo>
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
                                  f(x) = \beta_0 +
                                  \beta_1x_1 +
                                  \beta_2x_2 + \ldots
                                  + \beta_nx_n
                                </annotation>
                              </semantics>
                            </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal" style={{marginRight: '0.10764em'}}>f</span><span className="mopen">(</span><span className="mord mathnormal">x</span><span className="mclose">)</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">0</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">1</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">1</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight">2</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.6667em', verticalAlign: '-0.0833em'}} /><span className="minner">…</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.05278em'}}>β</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '-0.0528em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">n</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">n</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                    </p>
                    <p>For non-linear SVM, using kernel trick:
                      <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                              <semantics>
                                <mrow>
                                  <mi>K</mi>
                                  <mo stretchy="false">
                                    (</mo>
                                  <mi>x</mi>
                                  <mo separator="true">
                                    ,</mo>
                                  <mi>y</mi>
                                  <mo stretchy="false">
                                    )</mo>
                                  <mo>=</mo>
                                  <mi>ϕ</mi>
                                  <mo stretchy="false">
                                    (</mo>
                                  <mi>x</mi>
                                  <msup>
                                    <mo stretchy="false">
                                      )</mo>
                                    <mi>T</mi>
                                  </msup>
                                  <mi>ϕ</mi>
                                  <mo stretchy="false">
                                    (</mo>
                                  <mi>y</mi>
                                  <mo stretchy="false">
                                    )</mo>
                                </mrow>
                                <annotation encoding="application/x-tex">
                                  K(x, y) = \phi(x)^T
                                  \phi(y)</annotation>
                              </semantics>
                            </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal" style={{marginRight: '0.07153em'}}>K</span><span className="mopen">(</span><span className="mord mathnormal">x</span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span><span className="mclose">)</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.0913em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal">ϕ</span><span className="mopen">(</span><span className="mord mathnormal">x</span><span className="mclose"><span className="mclose">)</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8413em'}}><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.13889em'}}>T</span></span></span></span></span></span></span></span><span className="mord mathnormal">ϕ</span><span className="mopen">(</span><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>y</span><span className="mclose">)</span></span></span></span></span>
                    </p>
                    <p><strong>Implementation in Python:</strong>
                    </p>
                    <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> sklearn.svm <span className="hljs-keyword">import</span> SVC{"\n"}{"\n"}<span className="hljs-comment"># Create an SVM model with linear kernel</span>{"\n"}model = SVC(kernel=<span className="hljs-string">'linear'</span>){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train){"\n"}{"\n"}<span className="hljs-comment"># Make predictions</span>{"\n"}y_pred = model.predict(X_test){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}accuracy = accuracy_score(y_test, y_pred){"\n"}conf_matrix = confusion_matrix(y_test, y_pred){"\n"}class_report = classification_report(y_test, y_pred){"\n"}{"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Accuracy: <span className="hljs-subst">{"{"}accuracy{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">'Confusion Matrix:'</span>){"\n"}<span className="hljs-built_in">print</span>(conf_matrix){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">'Classification Report:'</span>){"\n"}<span className="hljs-built_in">print</span>(class_report){"\n"}</code></div></pre></div>
                  <h4>Decision Trees</h4>
                  <p>Decision trees are non-parametric models that
                    use a tree-like structure to make decisions
                    based on the values of the features. They
                    are easy to interpret and can handle both
                    numerical and categorical data.</p>
                  <p><strong>Algorithm:</strong></p>
                  <ol>
                    <li>Select the best feature to split the
                      data based on a criterion (e.g., Gini
                      impurity, information gain).</li>
                    <li>Split the data into subsets based on the
                      selected feature.</li>
                    <li>Repeat the process recursively for each
                      subset until a stopping criterion is
                      met.</li>
                  </ol>
                  <p><strong>Implementation in Python:</strong>
                  </p>
                  <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> sklearn.tree <span className="hljs-keyword">import</span> DecisionTreeClassifier{"\n"}<span className="hljs-keyword">from</span> sklearn.tree <span className="hljs-keyword">import</span> plot_tree{"\n"}{"\n"}<span className="hljs-comment"># Create a decision tree model</span>{"\n"}model = DecisionTreeClassifier(){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train){"\n"}{"\n"}<span className="hljs-comment"># Make predictions</span>{"\n"}y_pred = model.predict(X_test){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}accuracy = accuracy_score(y_test, y_pred){"\n"}conf_matrix = confusion_matrix(y_test, y_pred){"\n"}class_report = classification_report(y_test, y_pred){"\n"}{"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Accuracy: <span className="hljs-subst">{"{"}accuracy{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">'Confusion Matrix:'</span>){"\n"}<span className="hljs-built_in">print</span>(conf_matrix){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">'Classification Report:'</span>){"\n"}<span className="hljs-built_in">print</span>(class_report){"\n"}{"\n"}<span className="hljs-comment"># Plot the decision tree</span>{"\n"}plt.figure(figsize=(<span className="hljs-number">12</span>,<span className="hljs-number">8</span>)){"\n"}plot_tree(model, feature_names=[<span className="hljs-string">'feature1'</span>, <span className="hljs-string">'feature2'</span>, <span className="hljs-string">'feature3'</span>], class_names=[<span className="hljs-string">'class0'</span>, <span className="hljs-string">'class1'</span>], filled=<span className="hljs-literal">True</span>){"\n"}plt.show(){"\n"}</code></div></pre></div>
                <h4>Random Forests</h4>
                <p>Random forests are ensemble models that
                  consist of multiple decision trees. They
                  combine the predictions of individual trees
                  to improve accuracy and reduce overfitting.
                </p>
                <p><strong>Algorithm:</strong></p>
                <ol>
                  <li>Create multiple bootstrap samples from
                    the training data.</li>
                  <li>Train a decision tree on each bootstrap
                    sample.</li>
                  <li>Aggregate the predictions from all trees
                    (e.g., by majority voting for
                    classification).</li>
                </ol>
                <p><strong>Implementation in Python:</strong>
                </p>
                <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> sklearn.ensemble <span className="hljs-keyword">import</span> RandomForestClassifier{"\n"}{"\n"}<span className="hljs-comment"># Create a random forest model</span>{"\n"}model = RandomForestClassifier(n_estimators=<span className="hljs-number">100</span>, random_state=<span className="hljs-number">42</span>){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train){"\n"}{"\n"}<span className="hljs-comment"># Make predictions</span>{"\n"}y_pred = model.predict(X_test){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}accuracy = accuracy_score(y_test, y_pred){"\n"}conf_matrix = confusion_matrix(y_test, y_pred){"\n"}class_report = classification_report(y_test, y_pred){"\n"}{"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Accuracy: <span className="hljs-subst">{"{"}accuracy{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">'Confusion Matrix:'</span>){"\n"}<span className="hljs-built_in">print</span>(conf_matrix){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">'Classification Report:'</span>){"\n"}<span className="hljs-built_in">print</span>(class_report){"\n"}</code></div></pre></div>
              <h4>Model Evaluation Metrics</h4>
              <p>Evaluating classification models requires
                specific metrics to assess their performance
                accurately. Common evaluation metrics for
                classification models include:</p>
              <ol>
                <li>
                  <p><strong>Accuracy</strong>: The
                    proportion of correctly classified
                    instances.
                    <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                            <semantics>
                              <mrow>
                                <mtext>
                                  Accuracy
                                </mtext>
                                <mo>=</mo>
                                <mfrac>
                                  <mrow>
                                    <mtext>
                                      TP
                                    </mtext>
                                    <mo>+
                                    </mo>
                                    <mtext>
                                      TN
                                    </mtext>
                                  </mrow>
                                  <mrow>
                                    <mtext>
                                      TP
                                    </mtext>
                                    <mo>+
                                    </mo>
                                    <mtext>
                                      TN
                                    </mtext>
                                    <mo>+
                                    </mo>
                                    <mtext>
                                      FP
                                    </mtext>
                                    <mo>+
                                    </mo>
                                    <mtext>
                                      FN
                                    </mtext>
                                  </mrow>
                                </mfrac>
                              </mrow>
                              <annotation encoding="application/x-tex">
                                \text{'{'}Accuracy{'}'}
                                =
                                \frac{'{'}\text{'{'}TP{'}'}
                                +
                                \text{'{'}TN{'}'}{'}'}{'{'}\text{'{'}TP{'}'}
                                + \text{'{'}TN{'}'}
                                + \text{'{'}FP{'}'}
                                + \text{'{'}FN{'}'}{'}'}
                              </annotation>
                            </semantics>
                          </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8778em', verticalAlign: '-0.1944em'}} /><span className="mord text"><span className="mord">Accuracy</span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.2757em', verticalAlign: '-0.4033em'}} /><span className="mord"><span className="mopen nulldelimiter" /><span className="mfrac"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8723em'}}><span style={{top: '-2.655em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord text mtight"><span className="mord mtight">TP</span></span><span className="mbin mtight">+</span><span className="mord text mtight"><span className="mord mtight">TN</span></span><span className="mbin mtight">+</span><span className="mord text mtight"><span className="mord mtight">FP</span></span><span className="mbin mtight">+</span><span className="mord text mtight"><span className="mord mtight">FN</span></span></span></span></span><span style={{top: '-3.23em'}}><span className="pstrut" style={{height: '3em'}} /><span className="frac-line" style={{borderBottomWidth: '0.04em'}} /></span><span style={{top: '-3.394em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord text mtight"><span className="mord mtight">TP</span></span><span className="mbin mtight">+</span><span className="mord text mtight"><span className="mord mtight">TN</span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4033em'}}><span /></span></span></span></span><span className="mclose nulldelimiter" /></span></span></span></span></span>
                  </p>
                </li>
                <li>
                  <p><strong>Precision</strong>: The
                    proportion of true positive
                    instances among the instances
                    predicted as positive.
                    <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                            <semantics>
                              <mrow>
                                <mtext>
                                  Precision
                                </mtext>
                                <mo>=</mo>
                                <mfrac>
                                  <mtext>
                                    TP
                                  </mtext>
                                  <mrow>
                                    <mtext>
                                      TP
                                    </mtext>
                                    <mo>+
                                    </mo>
                                    <mtext>
                                      FP
                                    </mtext>
                                  </mrow>
                                </mfrac>
                              </mrow>
                              <annotation encoding="application/x-tex">
                                \text{'{'}Precision{'}'}
                                =
                                \frac{'{'}\text{'{'}TP{'}'}{'}'}{'{'}\text{'{'}TP{'}'}
                                + \text{'{'}FP{'}'}{'}'}
                              </annotation>
                            </semantics>
                          </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.6833em'}} /><span className="mord text"><span className="mord">Precision</span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.2757em', verticalAlign: '-0.4033em'}} /><span className="mord"><span className="mopen nulldelimiter" /><span className="mfrac"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8723em'}}><span style={{top: '-2.655em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord text mtight"><span className="mord mtight">TP</span></span><span className="mbin mtight">+</span><span className="mord text mtight"><span className="mord mtight">FP</span></span></span></span></span><span style={{top: '-3.23em'}}><span className="pstrut" style={{height: '3em'}} /><span className="frac-line" style={{borderBottomWidth: '0.04em'}} /></span><span style={{top: '-3.394em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord text mtight"><span className="mord mtight">TP</span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4033em'}}><span /></span></span></span></span><span className="mclose nulldelimiter" /></span></span></span></span></span>
                  </p>
                </li>
                <li>
                  <p><strong>Recall</strong>: The
                    proportion of true positive
                    instances among the actual positive
                    instances.
                    <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                            <semantics>
                              <mrow>
                                <mtext>
                                  Recall
                                </mtext>
                                <mo>=</mo>
                                <mfrac>
                                  <mtext>
                                    TP
                                  </mtext>
                                  <mrow>
                                    <mtext>
                                      TP
                                    </mtext>
                                    <mo>+
                                    </mo>
                                    <mtext>
                                      FN
                                    </mtext>
                                  </mrow>
                                </mfrac>
                              </mrow>
                              <annotation encoding="application/x-tex">
                                \text{'{'}Recall{'}'}
                                =
                                \frac{'{'}\text{'{'}TP{'}'}{'}'}{'{'}\text{'{'}TP{'}'}
                                + \text{'{'}FN{'}'}{'}'}
                              </annotation>
                            </semantics>
                          </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.6944em'}} /><span className="mord text"><span className="mord">Recall</span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.2757em', verticalAlign: '-0.4033em'}} /><span className="mord"><span className="mopen nulldelimiter" /><span className="mfrac"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8723em'}}><span style={{top: '-2.655em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord text mtight"><span className="mord mtight">TP</span></span><span className="mbin mtight">+</span><span className="mord text mtight"><span className="mord mtight">FN</span></span></span></span></span><span style={{top: '-3.23em'}}><span className="pstrut" style={{height: '3em'}} /><span className="frac-line" style={{borderBottomWidth: '0.04em'}} /></span><span style={{top: '-3.394em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord text mtight"><span className="mord mtight">TP</span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4033em'}}><span /></span></span></span></span><span className="mclose nulldelimiter" /></span></span></span></span></span>
                  </p>
                </li>
                <li>
                  <p><strong>F1 Score</strong>: The
                    harmonic mean of precision and
                    recall.
                    <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                            <semantics>
                              <mrow>
                                <mtext>
                                  F1&nbsp;Score
                                </mtext>
                                <mo>=</mo>
                                <mn>2</mn>
                                <mo>⋅</mo>
                                <mfrac>
                                  <mrow>
                                    <mtext>
                                      Precision
                                    </mtext>
                                    <mo>⋅
                                    </mo>
                                    <mtext>
                                      Recall
                                    </mtext>
                                  </mrow>
                                  <mrow>
                                    <mtext>
                                      Precision
                                    </mtext>
                                    <mo>+
                                    </mo>
                                    <mtext>
                                      Recall
                                    </mtext>
                                  </mrow>
                                </mfrac>
                              </mrow>
                              <annotation encoding="application/x-tex">
                                \text{'{'}F1
                                Score{'}'} = 2
                                \cdot
                                \frac{'{'}\text{'{'}Precision{'}'}
                                \cdot
                                \text{'{'}Recall{'}'}{'}'}{'{'}\text{'{'}Precision{'}'}
                                +
                                \text{'{'}Recall{'}'}{'}'}
                              </annotation>
                            </semantics>
                          </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.6833em'}} /><span className="mord text"><span className="mord">F1&nbsp;Score</span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '0.6444em'}} /><span className="mord">2</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">⋅</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1.2834em', verticalAlign: '-0.4033em'}} /><span className="mord"><span className="mopen nulldelimiter" /><span className="mfrac"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8801em'}}><span style={{top: '-2.655em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord text mtight"><span className="mord mtight">Precision</span></span><span className="mbin mtight">+</span><span className="mord text mtight"><span className="mord mtight">Recall</span></span></span></span></span><span style={{top: '-3.23em'}}><span className="pstrut" style={{height: '3em'}} /><span className="frac-line" style={{borderBottomWidth: '0.04em'}} /></span><span style={{top: '-3.394em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord text mtight"><span className="mord mtight">Precision</span></span><span className="mbin mtight">⋅</span><span className="mord text mtight"><span className="mord mtight">Recall</span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4033em'}}><span /></span></span></span></span><span className="mclose nulldelimiter" /></span></span></span></span></span>
                  </p>
                </li>
                <li>
                  <p><strong>Confusion Matrix</strong>: A
                    table that summarizes the
                    performance of a classification
                    model by showing the counts of true
                    positive (TP), true negative (TN),
                    false positive (FP), and false
                    negative (FN) instances.</p>
                </li>
              </ol>
              <p><strong>Implementation in Python:</strong>
              </p>
              <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> sklearn.metrics <span className="hljs-keyword">import</span> accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report{"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}accuracy = accuracy_score(y_test, y_pred){"\n"}precision = precision_score(y_test, y_pred, average=<span className="hljs-string">'weighted'</span>){"\n"}recall = recall_score(y_test, y_pred, average=<span className="hljs-string">'weighted'</span>){"\n"}f1 = f1_score(y_test, y_pred, average=<span className="hljs-string">'weighted'</span>){"\n"}conf_matrix = confusion_matrix(y_test, y_pred){"\n"}class_report = classification_report(y_test, y_pred){"\n"}{"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Accuracy: <span className="hljs-subst">{"{"}accuracy{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Precision: <span className="hljs-subst">{"{"}precision{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Recall: <span className="hljs-subst">{"{"}recall{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'F1 Score: <span className="hljs-subst">{"{"}f1{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">'Confusion Matrix:'</span>){"\n"}<span className="hljs-built_in">print</span>(conf_matrix){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">'Classification Report:'</span>){"\n"}<span className="hljs-built_in">print</span>(class_report){"\n"}</code></div></pre></div>
            <h4>Practical Tips for Classification</h4>
            <p>Here are some practical tips to improve your
              classification models:</p>
            <ol>
              <li><strong>Feature Engineering</strong>:
                Create new features or transform
                existing ones to capture complex
                relationships and interactions.</li>
              <li><strong>Feature Selection</strong>: Use
                techniques like forward selection,
                backward elimination, or regularization
                methods to select relevant features.
              </li>
              <li><strong>Handling Imbalanced
                  Data</strong>: Use techniques like
                oversampling, undersampling, or class
                weighting to address class imbalance.
              </li>
              <li><strong>Model Validation</strong>: Use
                cross-validation to assess the model's
                performance and prevent overfitting.
              </li>
              <li><strong>Hyperparameter Tuning</strong>:
                Optimize the model's hyperparameters
                using grid search or random search to
                improve performance.</li>
            </ol>
            <h4>Conclusion</h4>
            <p>Classification is a fundamental technique in
              supervised learning for predicting
              categorical outcomes. This chapter covered
              the basics of classification, including
              logistic regression, k-nearest neighbors,
              support vector machines, decision trees, and
              random forests. We also discussed model
              evaluation metrics and provided practical
              tips for improving classification models. By
              understanding and implementing these
              classification techniques, you can build
              robust models that accurately predict
              categorical outcomes and make informed
              decisions based on your data.</p>
          </div>
        </div>
      </div>
   )
}

export default ChapterSix;