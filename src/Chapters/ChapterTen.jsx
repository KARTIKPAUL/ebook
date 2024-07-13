import React from "react";

function ChapterTen(){
    return(
        <div className="w-full text-token-text-primary" dir="auto" id="ch-10" data-testid="conversation-turn-23" data-scroll-anchor="false">
        <div className="py-2 juice:py-[18px] px-3 text-base md:px-4 m-auto">
          <div className="mx-auto flex flex-1 gap-3 text-base juice:gap-4 juice:md:gap-6 md:max-w-3xl lg:max-w-[40rem] xl:max-w-[48rem]">
            <div className="group/conversation-turn relative flex w-full min-w-0 flex-col agent-turn">
              <div className="flex-col gap-1 md:gap-3">
                <div className="flex flex-grow flex-col max-w-full">
                  <div data-message-author-role="assistant" data-message-id="31794b57-55e3-4d68-a8c1-718309d70419" dir="auto" className="text-message flex flex-col items-start whitespace-pre-wrap break-words [.text-message+&]:mt-5 juice:w-full juice:items-end overflow-x-auto gap-2">
                    <div className="flex w-full flex-col gap-1 juice:empty:hidden juice:first:pt-[3px]">
                      <div className="markdown prose w-full break-words dark:prose-invert dark">
                        <h3>Chapter 10: Neural Networks and Deep
                          Learning</h3>
                        <hr />
                        <p>Neural networks and deep learning have
                          revolutionized the field of machine
                          learning, enabling the development of models
                          that can handle complex tasks such as image
                          recognition, natural language processing,
                          and game playing. This chapter explores the
                          fundamentals of neural networks, introduces
                          deep learning concepts, and demonstrates how
                          to build and train neural networks using
                          popular Python libraries like TensorFlow and
                          Keras.</p>
                        <h4>Introduction to Neural Networks</h4>
                        <p>Neural networks are computational models
                          inspired by the human brain's structure and
                          function. They consist of interconnected
                          layers of nodes, or neurons, that process
                          input data to produce an output. Neural
                          networks can learn complex patterns and
                          representations from data, making them
                          powerful tools for a wide range of tasks.
                        </p>
                        <p><strong>Key Concepts:</strong></p>
                        <ul>
                          <li><strong>Neuron</strong>: The basic unit
                            of a neural network, which receives
                            input, applies a weight and bias, and
                            produces an output through an activation
                            function.</li>
                          <li><strong>Layer</strong>: A collection of
                            neurons operating at the same depth
                            within the network. Layers can be input,
                            hidden, or output layers.</li>
                          <li><strong>Activation Function</strong>: A
                            function applied to a neuron's weighted
                            input to introduce non-linearity into
                            the network.</li>
                          <li><strong>Loss Function</strong>: A
                            function that measures the difference
                            between the predicted and actual output,
                            guiding the optimization process.</li>
                          <li><strong>Optimizer</strong>: An algorithm
                            that adjusts the network's weights and
                            biases to minimize the loss function.
                          </li>
                        </ul>
                        <h4>Structure of a Neural Network</h4>
                        <p>A neural network typically consists of the
                          following layers:</p>
                        <ol>
                          <li><strong>Input Layer</strong>: The first
                            layer that receives the input data.</li>
                          <li><strong>Hidden Layers</strong>:
                            Intermediate layers that process the
                            input data and learn complex patterns.
                            The number of hidden layers and neurons
                            in each layer can vary.</li>
                          <li><strong>Output Layer</strong>: The final
                            layer that produces the network's
                            output, corresponding to the task at
                            hand (e.g., classification, regression).
                          </li>
                        </ol>
                        <p><strong>Example of a Simple Neural
                            Network:</strong></p>
                        <pre><div className="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div className="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>text</span><div className="flex items-center"><span className data-state="closed"><button className="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width={24} height={24} fill="none" viewBox="0 0 24 24" className="icon-sm"><path fill="currentColor" fillRule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clipRule="evenodd" /></svg>Copy code</button></span></div></div><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-text">Input Layer -&gt; Hidden Layer -&gt; Output Layer{"\n"}</code></div></div></pre>
                        <p>Each connection between neurons has a weight
                          and a bias associated with it. During
                          training, these weights and biases are
                          adjusted to minimize the loss function.</p>
                        <h4>Activation Functions</h4>
                        <p>Activation functions introduce non-linearity
                          into the network, allowing it to learn
                          complex patterns. Common activation
                          functions include:</p>
                        <ol>
                          <li>
                            <p><strong>Sigmoid</strong>: Maps the
                              input to a value between 0 and 1.
                              <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                      <semantics>
                                        <mrow>
                                          <mi>σ</mi>
                                          <mo stretchy="false">
                                            (</mo>
                                          <mi>x</mi>
                                          <mo stretchy="false">
                                            )</mo>
                                          <mo>=</mo>
                                          <mfrac>
                                            <mn>1
                                            </mn>
                                            <mrow>
                                              <mn>1
                                              </mn>
                                              <mo>+
                                              </mo>
                                              <msup>
                                                <mi>e
                                                </mi>
                                                <mrow>
                                                  <mo>−
                                                  </mo>
                                                  <mi>x
                                                  </mi>
                                                </mrow>
                                              </msup>
                                            </mrow>
                                          </mfrac>
                                        </mrow>
                                        <annotation encoding="application/x-tex">
                                          \sigma(x) =
                                          \frac{'{'}1{'}'}{'{'}1 +
                                          e^{'{'}-x{'}'}{'}'}
                                        </annotation>
                                      </semantics>
                                    </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>σ</span><span className="mopen">(</span><span className="mord mathnormal">x</span><span className="mclose">)</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.2484em', verticalAlign: '-0.4033em'}} /><span className="mord"><span className="mopen nulldelimiter" /><span className="mfrac"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.8451em'}}><span style={{top: '-2.655em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight">1</span><span className="mbin mtight">+</span><span className="mord mtight"><span className="mord mathnormal mtight">e</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.7027em'}}><span style={{top: '-2.786em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mtight">−</span><span className="mord mathnormal mtight">x</span></span></span></span></span></span></span></span></span></span></span></span><span style={{top: '-3.23em'}}><span className="pstrut" style={{height: '3em'}} /><span className="frac-line" style={{borderBottomWidth: '0.04em'}} /></span><span style={{top: '-3.394em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4033em'}}><span /></span></span></span></span><span className="mclose nulldelimiter" /></span></span></span></span></span>
                            </p>
                          </li>
                          <li>
                            <p><strong>Tanh</strong>: Maps the input
                              to a value between -1 and 1.
                              <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                      <semantics>
                                        <mrow>
                                          <mi>tanh
                                          </mi>
                                          <mo>⁡</mo>
                                          <mo stretchy="false">
                                            (</mo>
                                          <mi>x</mi>
                                          <mo stretchy="false">
                                            )</mo>
                                          <mo>=</mo>
                                          <mfrac>
                                            <mrow>
                                              <msup>
                                                <mi>e
                                                </mi>
                                                <mi>x
                                                </mi>
                                              </msup>
                                              <mo>−
                                              </mo>
                                              <msup>
                                                <mi>e
                                                </mi>
                                                <mrow>
                                                  <mo>−
                                                  </mo>
                                                  <mi>x
                                                  </mi>
                                                </mrow>
                                              </msup>
                                            </mrow>
                                            <mrow>
                                              <msup>
                                                <mi>e
                                                </mi>
                                                <mi>x
                                                </mi>
                                              </msup>
                                              <mo>+
                                              </mo>
                                              <msup>
                                                <mi>e
                                                </mi>
                                                <mrow>
                                                  <mo>−
                                                  </mo>
                                                  <mi>x
                                                  </mi>
                                                </mrow>
                                              </msup>
                                            </mrow>
                                          </mfrac>
                                        </mrow>
                                        <annotation encoding="application/x-tex">
                                          \tanh(x) =
                                          \frac{'{'}e^x -
                                          e^{'{'}-x{'}'}{'}'}{'{'}e^x
                                          + e^{'{'}-x{'}'}{'}'}
                                        </annotation>
                                      </semantics>
                                    </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mop">tanh</span><span className="mopen">(</span><span className="mord mathnormal">x</span><span className="mclose">)</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.3907em', verticalAlign: '-0.4033em'}} /><span className="mord"><span className="mopen nulldelimiter" /><span className="mfrac"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.9874em'}}><span style={{top: '-2.655em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight"><span className="mord mathnormal mtight">e</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.5935em'}}><span style={{top: '-2.786em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mathnormal mtight">x</span></span></span></span></span></span></span></span><span className="mbin mtight">+</span><span className="mord mtight"><span className="mord mathnormal mtight">e</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.7027em'}}><span style={{top: '-2.786em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mtight">−</span><span className="mord mathnormal mtight">x</span></span></span></span></span></span></span></span></span></span></span></span><span style={{top: '-3.23em'}}><span className="pstrut" style={{height: '3em'}} /><span className="frac-line" style={{borderBottomWidth: '0.04em'}} /></span><span style={{top: '-3.394em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight"><span className="mord mathnormal mtight">e</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.7385em'}}><span style={{top: '-2.931em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mathnormal mtight">x</span></span></span></span></span></span></span></span><span className="mbin mtight">−</span><span className="mord mtight"><span className="mord mathnormal mtight">e</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8477em'}}><span style={{top: '-2.931em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mtight">−</span><span className="mord mathnormal mtight">x</span></span></span></span></span></span></span></span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4033em'}}><span /></span></span></span></span><span className="mclose nulldelimiter" /></span></span></span></span></span>
                            </p>
                          </li>
                          <li>
                            <p><strong>ReLU (Rectified Linear
                                Unit)</strong>: Replaces
                              negative values with zero.
                              <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                      <semantics>
                                        <mrow>
                                          <mtext>ReLU
                                          </mtext>
                                          <mo stretchy="false">
                                            (</mo>
                                          <mi>x</mi>
                                          <mo stretchy="false">
                                            )</mo>
                                          <mo>=</mo>
                                          <mi>max</mi>
                                          <mo>⁡</mo>
                                          <mo stretchy="false">
                                            (</mo>
                                          <mn>0</mn>
                                          <mo separator="true">
                                            ,</mo>
                                          <mi>x</mi>
                                          <mo stretchy="false">
                                            )</mo>
                                        </mrow>
                                        <annotation encoding="application/x-tex">
                                          \text{'{'}ReLU{'}'}(x)
                                          = \max(0, x)
                                        </annotation>
                                      </semantics>
                                    </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord text"><span className="mord">ReLU</span></span><span className="mopen">(</span><span className="mord mathnormal">x</span><span className="mclose">)</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mop">max</span><span className="mopen">(</span><span className="mord">0</span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord mathnormal">x</span><span className="mclose">)</span></span></span></span></span>
                            </p>
                          </li>
                          <li>
                            <p><strong>Leaky ReLU</strong>: Allows a
                              small gradient for negative values.
                              <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                      <semantics>
                                        <mrow>
                                          <mtext>
                                            Leaky&nbsp;ReLU
                                          </mtext>
                                          <mo stretchy="false">
                                            (</mo>
                                          <mi>x</mi>
                                          <mo stretchy="false">
                                            )</mo>
                                          <mo>=</mo>
                                          <mrow>
                                            <mo fence="true">
                                              {'{'}
                                            </mo>
                                            <mtable rowspacing="0.36em" columnalign="left left" columnspacing="1em">
                                              <mtr>
                                                <mtd>
                                                  <mstyle scriptlevel={0} displaystyle="false">
                                                    <mi>x
                                                    </mi>
                                                  </mstyle>
                                                </mtd>
                                                <mtd>
                                                  <mstyle scriptlevel={0} displaystyle="false">
                                                    <mrow>
                                                      <mtext>
                                                        if&nbsp;
                                                      </mtext>
                                                      <mi>x
                                                      </mi>
                                                      <mo>&gt;
                                                      </mo>
                                                      <mn>0
                                                      </mn>
                                                    </mrow>
                                                  </mstyle>
                                                </mtd>
                                              </mtr>
                                              <mtr>
                                                <mtd>
                                                  <mstyle scriptlevel={0} displaystyle="false">
                                                    <mrow>
                                                      <mi>α
                                                      </mi>
                                                      <mi>x
                                                      </mi>
                                                    </mrow>
                                                  </mstyle>
                                                </mtd>
                                                <mtd>
                                                  <mstyle scriptlevel={0} displaystyle="false">
                                                    <mtext>
                                                      otherwise
                                                    </mtext>
                                                  </mstyle>
                                                </mtd>
                                              </mtr>
                                            </mtable>
                                          </mrow>
                                        </mrow>
                                        <annotation encoding="application/x-tex">
                                          \text{'{'}Leaky
                                          ReLU{'}'}(x) =
                                          \begin{'{'}cases{'}'}
                                          x &amp;
                                          \text{'{'}if {'}'} x
                                          &gt; 0 \\
                                          \alpha x
                                          &amp;
                                          \text{'{'}otherwise{'}'}
                                          \end{'{'}cases{'}'}
                                        </annotation>
                                      </semantics>
                                    </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord text"><span className="mord">Leaky&nbsp;ReLU</span></span><span className="mopen">(</span><span className="mord mathnormal">x</span><span className="mclose">)</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '3em', verticalAlign: '-1.25em'}} /><span className="minner"><span className="mopen delimcenter" style={{top: '0em'}}><span className="delimsizing size4">{'{'}</span></span><span className="mord"><span className="mtable"><span className="col-align-l"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '1.69em'}}><span style={{top: '-3.69em'}}><span className="pstrut" style={{height: '3.008em'}} /><span className="mord"><span className="mord mathnormal">x</span></span></span><span style={{top: '-2.25em'}}><span className="pstrut" style={{height: '3.008em'}} /><span className="mord"><span className="mord mathnormal">αx</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '1.19em'}}><span /></span></span></span></span><span className="arraycolsep" style={{width: '1em'}} /><span className="col-align-l"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '1.69em'}}><span style={{top: '-3.69em'}}><span className="pstrut" style={{height: '3.008em'}} /><span className="mord"><span className="mord text"><span className="mord">if&nbsp;</span></span><span className="mord mathnormal">x</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">&gt;</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mord">0</span></span></span><span style={{top: '-2.25em'}}><span className="pstrut" style={{height: '3.008em'}} /><span className="mord"><span className="mord text"><span className="mord">otherwise</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '1.19em'}}><span /></span></span></span></span></span></span><span className="mclose nulldelimiter" /></span></span></span></span></span>
                            </p>
                          </li>
                        </ol>
                        <h4>Building a Neural Network with Keras</h4>
                        <p>Keras is a high-level neural networks API,
                          written in Python and capable of running on
                          top of TensorFlow. It simplifies the process
                          of building and training neural networks.
                        </p>
                        <p><strong>Example: Building a Neural Network
                            for Classification</strong></p>
                        <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> numpy <span className="hljs-keyword">as</span> np{"\n"}<span className="hljs-keyword">import</span> pandas <span className="hljs-keyword">as</span> pd{"\n"}<span className="hljs-keyword">import</span> tensorflow <span className="hljs-keyword">as</span> tf{"\n"}<span className="hljs-keyword">from</span> tensorflow.keras <span className="hljs-keyword">import</span> layers, models{"\n"}<span className="hljs-keyword">from</span> sklearn.model_selection <span className="hljs-keyword">import</span> train_test_split{"\n"}<span className="hljs-keyword">from</span> sklearn.preprocessing <span className="hljs-keyword">import</span> StandardScaler{"\n"}{"\n"}<span className="hljs-comment"># Load dataset</span>{"\n"}data = pd.read_csv(<span className="hljs-string">'data.csv'</span>){"\n"}{"\n"}<span className="hljs-comment"># Define feature and target variables</span>{"\n"}X = data.drop(<span className="hljs-string">'target'</span>, axis=<span className="hljs-number">1</span>).values{"\n"}y = data[<span className="hljs-string">'target'</span>].values{"\n"}{"\n"}<span className="hljs-comment"># Split data into training and test sets</span>{"\n"}X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=<span className="hljs-number">0.3</span>, random_state=<span className="hljs-number">42</span>){"\n"}{"\n"}<span className="hljs-comment"># Standardize the data</span>{"\n"}scaler = StandardScaler(){"\n"}X_train = scaler.fit_transform(X_train){"\n"}X_test = scaler.transform(X_test){"\n"}{"\n"}<span className="hljs-comment"># Build the neural network</span>{"\n"}model = models.Sequential(){"\n"}model.add(layers.Dense(<span className="hljs-number">64</span>, activation=<span className="hljs-string">'relu'</span>, input_shape=(X_train.shape[<span className="hljs-number">1</span>],))){"\n"}model.add(layers.Dense(<span className="hljs-number">32</span>, activation=<span className="hljs-string">'relu'</span>)){"\n"}model.add(layers.Dense(<span className="hljs-number">1</span>, activation=<span className="hljs-string">'sigmoid'</span>)){"\n"}{"\n"}<span className="hljs-comment"># Compile the model</span>{"\n"}model.<span className="hljs-built_in">compile</span>(optimizer=<span className="hljs-string">'adam'</span>, loss=<span className="hljs-string">'binary_crossentropy'</span>, metrics=[<span className="hljs-string">'accuracy'</span>]){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train, epochs=<span className="hljs-number">20</span>, batch_size=<span className="hljs-number">32</span>, validation_split=<span className="hljs-number">0.2</span>){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}loss, accuracy = model.evaluate(X_test, y_test){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Loss: <span className="hljs-subst">{"{"}loss{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Accuracy: <span className="hljs-subst">{"{"}accuracy{"}"}</span>'</span>){"\n"}</code></div></pre></div>
                      <h4>Deep Learning</h4>
                      <p>Deep learning is a subset of machine learning
                        that involves neural networks with many
                        layers (deep neural networks). It can
                        automatically learn features from raw data,
                        making it especially powerful for tasks like
                        image recognition and natural language
                        processing.</p>
                      <p><strong>Deep Neural Network
                          Structure:</strong></p>
                      <pre><div className="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div className="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>text</span><div className="flex items-center"><span className data-state="closed"><button className="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width={24} height={24} fill="none" viewBox="0 0 24 24" className="icon-sm"><path fill="currentColor" fillRule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clipRule="evenodd" /></svg>Copy code</button></span></div></div><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-text">Input Layer -&gt; Hidden Layer 1 -&gt; Hidden Layer 2 -&gt; ... -&gt; Hidden Layer n -&gt; Output Layer{"\n"}</code></div></div></pre>
                      <h4>Convolutional Neural Networks (CNNs)</h4>
                      <p>CNNs are specialized neural networks designed
                        for processing structured grid data, such as
                        images. They use convolutional layers to
                        automatically learn spatial hierarchies of
                        features from input images.</p>
                      <p><strong>Key Components of CNNs:</strong></p>
                      <ol>
                        <li><strong>Convolutional Layers</strong>:
                          Apply convolution operations to the
                          input, extracting features like edges,
                          textures, and shapes.</li>
                        <li><strong>Pooling Layers</strong>: Reduce
                          the spatial dimensions of the feature
                          maps, retaining important information
                          while reducing computational complexity.
                        </li>
                        <li><strong>Fully Connected Layers</strong>:
                          Perform high-level reasoning based on
                          the extracted features.</li>
                      </ol>
                      <p><strong>Example: Building a CNN for Image
                          Classification</strong></p>
                      <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> tensorflow.keras.datasets <span className="hljs-keyword">import</span> mnist{"\n"}<span className="hljs-keyword">from</span> tensorflow.keras.utils <span className="hljs-keyword">import</span> to_categorical{"\n"}{"\n"}<span className="hljs-comment"># Load the dataset</span>{"\n"}(X_train, y_train), (X_test, y_test) = mnist.load_data(){"\n"}{"\n"}<span className="hljs-comment"># Preprocess the data</span>{"\n"}X_train = X_train.reshape((X_train.shape[<span className="hljs-number">0</span>], <span className="hljs-number">28</span>, <span className="hljs-number">28</span>, <span className="hljs-number">1</span>)).astype(<span className="hljs-string">'float32'</span>) / <span className="hljs-number">255</span>{"\n"}X_test = X_test.reshape((X_test.shape[<span className="hljs-number">0</span>], <span className="hljs-number">28</span>, <span className="hljs-number">28</span>, <span className="hljs-number">1</span>)).astype(<span className="hljs-string">'float32'</span>) / <span className="hljs-number">255</span>{"\n"}y_train = to_categorical(y_train){"\n"}y_test = to_categorical(y_test){"\n"}{"\n"}<span className="hljs-comment"># Build the CNN</span>{"\n"}model = models.Sequential(){"\n"}model.add(layers.Conv2D(<span className="hljs-number">32</span>, (<span className="hljs-number">3</span>, <span className="hljs-number">3</span>), activation=<span className="hljs-string">'relu'</span>, input_shape=(<span className="hljs-number">28</span>, <span className="hljs-number">28</span>, <span className="hljs-number">1</span>))){"\n"}model.add(layers.MaxPooling2D((<span className="hljs-number">2</span>, <span className="hljs-number">2</span>))){"\n"}model.add(layers.Conv2D(<span className="hljs-number">64</span>, (<span className="hljs-number">3</span>, <span className="hljs-number">3</span>), activation=<span className="hljs-string">'relu'</span>)){"\n"}model.add(layers.MaxPooling2D((<span className="hljs-number">2</span>, <span className="hljs-number">2</span>))){"\n"}model.add(layers.Conv2D(<span className="hljs-number">64</span>, (<span className="hljs-number">3</span>, <span className="hljs-number">3</span>), activation=<span className="hljs-string">'relu'</span>)){"\n"}model.add(layers.Flatten()){"\n"}model.add(layers.Dense(<span className="hljs-number">64</span>, activation=<span className="hljs-string">'relu'</span>)){"\n"}model.add(layers.Dense(<span className="hljs-number">10</span>, activation=<span className="hljs-string">'softmax'</span>)){"\n"}{"\n"}<span className="hljs-comment"># Compile the model</span>{"\n"}model.<span className="hljs-built_in">compile</span>(optimizer=<span className="hljs-string">'adam'</span>, loss=<span className="hljs-string">'categorical_crossentropy'</span>, metrics=[<span className="hljs-string">'accuracy'</span>]){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train, epochs=<span className="hljs-number">5</span>, batch_size=<span className="hljs-number">64</span>, validation_split=<span className="hljs-number">0.2</span>){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}loss, accuracy = model.evaluate(X_test, y_test){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Loss: <span className="hljs-subst">{"{"}loss{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Accuracy: <span className="hljs-subst">{"{"}accuracy{"}"}</span>'</span>){"\n"}</code></div></pre></div>
                    <h4>Recurrent Neural Networks (RNNs)</h4>
                    <p>RNNs are specialized neural networks designed
                      for sequential data, such as time series or
                      natural language. They use loops within the
                      network to maintain a memory of previous
                      inputs, making them suitable for tasks like
                      language modeling and speech recognition.
                    </p>
                    <p><strong>Key Components of RNNs:</strong></p>
                    <ol>
                      <li><strong>Recurrent Layers</strong>:
                        Maintain a hidden state that captures
                        information from previous time steps.
                      </li>
                      <li><strong>LSTM (Long Short-Term
                          Memory)</strong>: A type of RNN that
                        can capture long-term dependencies by
                        using special gating mechanisms to
                        control the flow of information.</li>
                    </ol>
                    <p><strong>Example: Building an LSTM for
                        Sequence Prediction</strong></p>
                    <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> tensorflow.keras.preprocessing <span className="hljs-keyword">import</span> sequence{"\n"}<span className="hljs-keyword">from</span> tensorflow.keras.datasets <span className="hljs-keyword">import</span> imdb{"\n"}{"\n"}<span className="hljs-comment"># Load the dataset</span>{"\n"}max_features = <span className="hljs-number">20000</span>{"\n"}max_len = <span className="hljs-number">100</span>{"\n"}(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features){"\n"}{"\n"}<span className="hljs-comment"># Preprocess the data</span>{"\n"}X_train = sequence.pad_sequences(X_train, maxlen=max_len){"\n"}X_test = sequence.pad_sequences(X_test, maxlen=max_len){"\n"}{"\n"}<span className="hljs-comment"># Build the LSTM</span>{"\n"}model = models.Sequential(){"\n"}model.add(layers.Embedding(max_features, <span className="hljs-number">128</span>, input_length=max_len)){"\n"}model.add(layers.LSTM(<span className="hljs-number">64</span>)){"\n"}model.add(layers.Dense(<span className="hljs-number">1</span>, activation=<span className="hljs-string">'sigmoid'</span>)){"\n"}{"\n"}<span className="hljs-comment"># Compile the model</span>{"\n"}model.<span className="hljs-built_in">compile</span>(optimizer=<span className="hljs-string">'adam'</span>, loss=<span className="hljs-string">'binary_crossentropy'</span>, metrics=[<span className="hljs-string">'accuracy'</span>]){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train, epochs=<span className="hljs-number">3</span>, batch_size=<span className="hljs-number">64</span>, validation_split=<span className="hljs-number">0.2</span>){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the model</span>{"\n"}loss, accuracy = model.evaluate(X_test, y_test){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Loss: <span className="hljs-subst">{"{"}loss{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Accuracy: <span className="hljs-subst">{"{"}accuracy{"}"}</span>'</span>){"\n"}</code></div></pre></div>
                  <h4>Practical Tips for Building Neural Networks
                  </h4>
                  <p>Here are some practical tips to improve your
                    neural network models:</p>
                  <ol>
                    <li><strong>Choose the Right
                        Architecture</strong>: Select the
                      appropriate network architecture (e.g.,
                      CNN, RNN) based on the nature of your
                      data and task.</li>
                    <li><strong>Use Regularization</strong>:
                      Apply techniques like dropout, weight
                      decay, and batch normalization to
                      prevent overfitting.</li>
                    <li><strong>Optimize
                        Hyperparameters</strong>: Tune
                      hyperparameters such as learning rate,
                      batch size, and number of epochs to
                      achieve better performance.</li>
                    <li><strong>Data Augmentation</strong>: Use
                      data augmentation techniques to increase
                      the diversity of your training data and
                      improve generalization.</li>
                    <li><strong>Monitor Training</strong>: Use
                      tools like TensorBoard to visualize and
                      monitor the training process, adjusting
                      parameters as needed.</li>
                  </ol>
                  <h4>Conclusion</h4>
                  <p>Neural networks and deep learning have
                    transformed the field of machine learning,
                    enabling the development of models that can
                    handle complex and diverse tasks. This
                    chapter covered the fundamentals of neural
                    networks, deep learning concepts, and
                    practical implementations using Python and
                    libraries like TensorFlow and Keras. By
                    understanding and applying these techniques,
                    you can build powerful models that can solve
                    a wide range of real-world problems and make
                    informed decisions based on your data.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
}

export default ChapterTen;