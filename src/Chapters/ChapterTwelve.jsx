import React from "react";

function ChapterTwelve(){
    return(
        <div>
        <div className="w-full text-token-text-primary" dir="auto" id="ch-12" data-testid="conversation-turn-27" data-scroll-anchor="false">
          <div className="py-2 juice:py-[18px] px-3 text-base md:px-4 m-auto">
            <div className="mx-auto flex flex-1 gap-3 text-base juice:gap-4 juice:md:gap-6 md:max-w-3xl lg:max-w-[40rem] xl:max-w-[48rem]">
              <div className="group/conversation-turn relative flex w-full min-w-0 flex-col agent-turn">
                <div className="flex-col gap-1 md:gap-3">
                  <div className="flex flex-grow flex-col max-w-full">
                    <div data-message-author-role="assistant" data-message-id="b609b380-fb25-45db-828c-1cc550037ac4" dir="auto" className="text-message flex flex-col items-start whitespace-pre-wrap break-words [.text-message+&]:mt-5 juice:w-full juice:items-end overflow-x-auto gap-2">
                      <div className="flex w-full flex-col gap-1 juice:empty:hidden juice:first:pt-[3px]">
                        <div className="markdown prose w-full break-words dark:prose-invert dark">
                          <h3>Chapter 12: Recurrent Neural Networks (RNNs)
                          </h3>
                          <hr />
                          <p>Recurrent Neural Networks (RNNs) are a class
                            of neural networks designed for sequential
                            data, making them well-suited for tasks
                            involving time series, natural language
                            processing, and other data with temporal
                            dependencies. This chapter delves into the
                            fundamentals of RNNs, their key components,
                            and how to implement and train them using
                            Python and TensorFlow/Keras.</p>
                          <h4>Introduction to Recurrent Neural Networks
                          </h4>
                          <p>Unlike feedforward neural networks, RNNs have
                            connections that form directed cycles,
                            allowing them to maintain a hidden state
                            that captures information from previous time
                            steps. This enables RNNs to process
                            sequences of data and learn temporal
                            dependencies.</p>
                          <p><strong>Key Concepts:</strong></p>
                          <ul>
                            <li><strong>Hidden State</strong>: A state
                              that carries information from previous
                              time steps to influence the current
                              output.</li>
                            <li><strong>Sequence</strong>: An ordered
                              set of data points.</li>
                            <li><strong>Vanishing Gradient
                                Problem</strong>: A challenge in
                              training RNNs where gradients diminish
                              as they are propagated back through
                              time, making it difficult to learn
                              long-range dependencies.</li>
                          </ul>
                          <h4>Structure of an RNN</h4>
                          <p>An RNN consists of units (neurons) that take
                            input at each time step and maintain a
                            hidden state. The hidden state is updated
                            based on the current input and the previous
                            hidden state.</p>
                          <p><strong>Basic RNN Equation:</strong></p>
                          <p><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                    <semantics>
                                      <mrow>
                                        <msub>
                                          <mi>h</mi>
                                          <mi>t</mi>
                                        </msub>
                                        <mo>=</mo>
                                        <mi>σ</mi>
                                        <mo stretchy="false">
                                          (</mo>
                                        <msub>
                                          <mi>W</mi>
                                          <mrow>
                                            <mi>h</mi>
                                            <mi>x</mi>
                                          </mrow>
                                        </msub>
                                        <msub>
                                          <mi>x</mi>
                                          <mi>t</mi>
                                        </msub>
                                        <mo>+</mo>
                                        <msub>
                                          <mi>W</mi>
                                          <mrow>
                                            <mi>h</mi>
                                            <mi>h</mi>
                                          </mrow>
                                        </msub>
                                        <msub>
                                          <mi>h</mi>
                                          <mrow>
                                            <mi>t</mi>
                                            <mo>−</mo>
                                            <mn>1</mn>
                                          </mrow>
                                        </msub>
                                        <mo>+</mo>
                                        <msub>
                                          <mi>b</mi>
                                          <mi>h</mi>
                                        </msub>
                                        <mo stretchy="false">
                                          )</mo>
                                      </mrow>
                                      <annotation encoding="application/x-tex">
                                        h_t =
                                        \sigma(W_{'{'}hx{'}'}x_t +
                                        W_{'{'}hh{'}'}h_{'{'}t-1{'}'} + b_h)
                                      </annotation>
                                    </semantics>
                                  </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8444em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>σ</span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>W</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3361em'}}><span style={{top: '-2.55em', marginLeft: '-0.1389em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">h</span><span className="mord mathnormal mtight">x</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.9028em', verticalAlign: '-0.2083em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>W</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3361em'}}><span style={{top: '-2.55em', marginLeft: '-0.1389em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">hh</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mbin mtight">−</span><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2083em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord"><span className="mord mathnormal">b</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3361em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">h</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">)</span></span></span></span></span>
                          </p>
                          <p>Where:</p>
                          <ul>
                            <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                      <semantics>
                                        <mrow>
                                          <msub>
                                            <mi>h</mi>
                                            <mi>t</mi>
                                          </msub>
                                        </mrow>
                                        <annotation encoding="application/x-tex">
                                          h_t</annotation>
                                      </semantics>
                                    </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8444em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                              is the hidden state at time step <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                      <semantics>
                                        <mrow>
                                          <mi>t</mi>
                                        </mrow>
                                        <annotation encoding="application/x-tex">
                                          t</annotation>
                                      </semantics>
                                    </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.6151em'}} /><span className="mord mathnormal">t</span></span></span></span></span>.
                            </li>
                            <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                      <semantics>
                                        <mrow>
                                          <msub>
                                            <mi>x</mi>
                                            <mi>t</mi>
                                          </msub>
                                        </mrow>
                                        <annotation encoding="application/x-tex">
                                          x_t</annotation>
                                      </semantics>
                                    </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.5806em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                              is the input at time step <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                      <semantics>
                                        <mrow>
                                          <mi>t</mi>
                                        </mrow>
                                        <annotation encoding="application/x-tex">
                                          t</annotation>
                                      </semantics>
                                    </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.6151em'}} /><span className="mord mathnormal">t</span></span></span></span></span>.
                            </li>
                            <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                      <semantics>
                                        <mrow>
                                          <msub>
                                            <mi>W</mi>
                                            <mrow>
                                              <mi>h
                                              </mi>
                                              <mi>x
                                              </mi>
                                            </mrow>
                                          </msub>
                                        </mrow>
                                        <annotation encoding="application/x-tex">
                                          W_{'{'}hx{'}'}
                                        </annotation>
                                      </semantics>
                                    </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8333em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>W</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3361em'}}><span style={{top: '-2.55em', marginLeft: '-0.1389em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">h</span><span className="mord mathnormal mtight">x</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                              and <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                      <semantics>
                                        <mrow>
                                          <msub>
                                            <mi>W</mi>
                                            <mrow>
                                              <mi>h
                                              </mi>
                                              <mi>h
                                              </mi>
                                            </mrow>
                                          </msub>
                                        </mrow>
                                        <annotation encoding="application/x-tex">
                                          W_{'{'}hh{'}'}
                                        </annotation>
                                      </semantics>
                                    </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8333em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>W</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3361em'}}><span style={{top: '-2.55em', marginLeft: '-0.1389em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">hh</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                              are weight matrices.</li>
                            <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                      <semantics>
                                        <mrow>
                                          <msub>
                                            <mi>b</mi>
                                            <mi>h</mi>
                                          </msub>
                                        </mrow>
                                        <annotation encoding="application/x-tex">
                                          b_h</annotation>
                                      </semantics>
                                    </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8444em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal">b</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3361em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">h</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                              is the bias term.</li>
                            <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                      <semantics>
                                        <mrow>
                                          <mi>σ</mi>
                                        </mrow>
                                        <annotation encoding="application/x-tex">
                                          \sigma
                                        </annotation>
                                      </semantics>
                                    </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.4306em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>σ</span></span></span></span></span>
                              is the activation function (e.g., tanh
                              or ReLU).</li>
                          </ul>
                          <h4>Long Short-Term Memory (LSTM)</h4>
                          <p>LSTMs are a type of RNN designed to address
                            the vanishing gradient problem. They use
                            gating mechanisms to control the flow of
                            information, allowing them to capture
                            long-range dependencies.</p>
                          <p><strong>Key Components of LSTM:</strong></p>
                          <ol>
                            <li>
                              <p><strong>Forget Gate</strong>: Decides
                                what information to discard from the
                                cell state.
                                <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                        <semantics>
                                          <mrow>
                                            <msub>
                                              <mi>f
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo>=</mo>
                                            <mi>σ</mi>
                                            <mo stretchy="false">
                                              (</mo>
                                            <msub>
                                              <mi>W
                                              </mi>
                                              <mi>f
                                              </mi>
                                            </msub>
                                            <mo>⋅</mo>
                                            <mo stretchy="false">
                                              [</mo>
                                            <msub>
                                              <mi>h
                                              </mi>
                                              <mrow>
                                                <mi>t
                                                </mi>
                                                <mo>−
                                                </mo>
                                                <mn>1
                                                </mn>
                                              </mrow>
                                            </msub>
                                            <mo separator="true">
                                              ,</mo>
                                            <msub>
                                              <mi>x
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo stretchy="false">
                                              ]</mo>
                                            <mo>+</mo>
                                            <msub>
                                              <mi>b
                                              </mi>
                                              <mi>f
                                              </mi>
                                            </msub>
                                            <mo stretchy="false">
                                              )</mo>
                                          </mrow>
                                          <annotation encoding="application/x-tex">
                                            f_t =
                                            \sigma(W_f
                                            \cdot
                                            [h_{'{'}t-1{'}'},
                                            x_t] + b_f)
                                          </annotation>
                                        </semantics>
                                      </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.10764em'}}>f</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '-0.1076em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.0361em', verticalAlign: '-0.2861em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>σ</span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>W</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3361em'}}><span style={{top: '-2.55em', marginLeft: '-0.1389em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.10764em'}}>f</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2861em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">⋅</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mopen">[</span><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mbin mtight">−</span><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2083em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">]</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1.0361em', verticalAlign: '-0.2861em'}} /><span className="mord"><span className="mord mathnormal">b</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3361em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.10764em'}}>f</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2861em'}}><span /></span></span></span></span></span><span className="mclose">)</span></span></span></span></span>
                              </p>
                            </li>
                            <li>
                              <p><strong>Input Gate</strong>: Decides
                                what new information to store in the
                                cell state.
                                <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                        <semantics>
                                          <mrow>
                                            <msub>
                                              <mi>i
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo>=</mo>
                                            <mi>σ</mi>
                                            <mo stretchy="false">
                                              (</mo>
                                            <msub>
                                              <mi>W
                                              </mi>
                                              <mi>i
                                              </mi>
                                            </msub>
                                            <mo>⋅</mo>
                                            <mo stretchy="false">
                                              [</mo>
                                            <msub>
                                              <mi>h
                                              </mi>
                                              <mrow>
                                                <mi>t
                                                </mi>
                                                <mo>−
                                                </mo>
                                                <mn>1
                                                </mn>
                                              </mrow>
                                            </msub>
                                            <mo separator="true">
                                              ,</mo>
                                            <msub>
                                              <mi>x
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo stretchy="false">
                                              ]</mo>
                                            <mo>+</mo>
                                            <msub>
                                              <mi>b
                                              </mi>
                                              <mi>i
                                              </mi>
                                            </msub>
                                            <mo stretchy="false">
                                              )</mo>
                                          </mrow>
                                          <annotation encoding="application/x-tex">
                                            i_t =
                                            \sigma(W_i
                                            \cdot
                                            [h_{'{'}t-1{'}'},
                                            x_t] + b_i)
                                          </annotation>
                                        </semantics>
                                      </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8095em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal">i</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>σ</span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>W</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '-0.1389em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">i</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">⋅</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mopen">[</span><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mbin mtight">−</span><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2083em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">]</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord"><span className="mord mathnormal">b</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3117em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">i</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">)</span></span></span></span></span>
                                <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                        <semantics>
                                          <mrow>
                                            <msub>
                                              <mover accent="true">
                                                <mi>C
                                                </mi>
                                                <mo>~
                                                </mo>
                                              </mover>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo>=</mo>
                                            <mi>tanh
                                            </mi>
                                            <mo>⁡</mo>
                                            <mo stretchy="false">
                                              (</mo>
                                            <msub>
                                              <mi>W
                                              </mi>
                                              <mi>C
                                              </mi>
                                            </msub>
                                            <mo>⋅</mo>
                                            <mo stretchy="false">
                                              [</mo>
                                            <msub>
                                              <mi>h
                                              </mi>
                                              <mrow>
                                                <mi>t
                                                </mi>
                                                <mo>−
                                                </mo>
                                                <mn>1
                                                </mn>
                                              </mrow>
                                            </msub>
                                            <mo separator="true">
                                              ,</mo>
                                            <msub>
                                              <mi>x
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo stretchy="false">
                                              ]</mo>
                                            <mo>+</mo>
                                            <msub>
                                              <mi>b
                                              </mi>
                                              <mi>C
                                              </mi>
                                            </msub>
                                            <mo stretchy="false">
                                              )</mo>
                                          </mrow>
                                          <annotation encoding="application/x-tex">
                                            \tilde{'{'}C{'}'}_t
                                            = \tanh(W_C
                                            \cdot
                                            [h_{'{'}t-1{'}'},
                                            x_t] + b_C)
                                          </annotation>
                                        </semantics>
                                      </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1.0702em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord accent"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.9202em'}}><span style={{top: '-3em'}}><span className="pstrut" style={{height: '3em'}} /><span className="mord mathnormal" style={{marginRight: '0.07153em'}}>C</span></span><span style={{top: '-3.6023em'}}><span className="pstrut" style={{height: '3em'}} /><span className="accent-body" style={{left: '-0.1667em'}}><span className="mord">~</span></span></span></span></span></span></span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '-0.0715em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mop">tanh</span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>W</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3283em'}}><span style={{top: '-2.55em', marginLeft: '-0.1389em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.07153em'}}>C</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">⋅</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mopen">[</span><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mbin mtight">−</span><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2083em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">]</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord"><span className="mord mathnormal">b</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3283em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.07153em'}}>C</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">)</span></span></span></span></span>
                              </p>
                            </li>
                            <li>
                              <p><strong>Cell State</strong>: The
                                memory of the network.
                                <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                        <semantics>
                                          <mrow>
                                            <msub>
                                              <mi>C
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo>=</mo>
                                            <msub>
                                              <mi>f
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo>∗</mo>
                                            <msub>
                                              <mi>C
                                              </mi>
                                              <mrow>
                                                <mi>t
                                                </mi>
                                                <mo>−
                                                </mo>
                                                <mn>1
                                                </mn>
                                              </mrow>
                                            </msub>
                                            <mo>+</mo>
                                            <msub>
                                              <mi>i
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo>∗</mo>
                                            <msub>
                                              <mover accent="true">
                                                <mi>C
                                                </mi>
                                                <mo>~
                                                </mo>
                                              </mover>
                                              <mi>t
                                              </mi>
                                            </msub>
                                          </mrow>
                                          <annotation encoding="application/x-tex">
                                            C_t = f_t *
                                            C_{'{'}t-1{'}'} +
                                            i_t *
                                            \tilde{'{'}C{'}'}_t
                                          </annotation>
                                        </semantics>
                                      </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8333em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.07153em'}}>C</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '-0.0715em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '0.8889em', verticalAlign: '-0.1944em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.10764em'}}>f</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '-0.1076em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">∗</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8917em', verticalAlign: '-0.2083em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.07153em'}}>C</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '-0.0715em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mbin mtight">−</span><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2083em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.8095em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal">i</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">∗</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1.0702em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord accent"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.9202em'}}><span style={{top: '-3em'}}><span className="pstrut" style={{height: '3em'}} /><span className="mord mathnormal" style={{marginRight: '0.07153em'}}>C</span></span><span style={{top: '-3.6023em'}}><span className="pstrut" style={{height: '3em'}} /><span className="accent-body" style={{left: '-0.1667em'}}><span className="mord">~</span></span></span></span></span></span></span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '-0.0715em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                              </p>
                            </li>
                            <li>
                              <p><strong>Output Gate</strong>: Decides
                                what information to output.
                                <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                        <semantics>
                                          <mrow>
                                            <msub>
                                              <mi>o
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo>=</mo>
                                            <mi>σ</mi>
                                            <mo stretchy="false">
                                              (</mo>
                                            <msub>
                                              <mi>W
                                              </mi>
                                              <mi>o
                                              </mi>
                                            </msub>
                                            <mo>⋅</mo>
                                            <mo stretchy="false">
                                              [</mo>
                                            <msub>
                                              <mi>h
                                              </mi>
                                              <mrow>
                                                <mi>t
                                                </mi>
                                                <mo>−
                                                </mo>
                                                <mn>1
                                                </mn>
                                              </mrow>
                                            </msub>
                                            <mo separator="true">
                                              ,</mo>
                                            <msub>
                                              <mi>x
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo stretchy="false">
                                              ]</mo>
                                            <mo>+</mo>
                                            <msub>
                                              <mi>b
                                              </mi>
                                              <mi>o
                                              </mi>
                                            </msub>
                                            <mo stretchy="false">
                                              )</mo>
                                          </mrow>
                                          <annotation encoding="application/x-tex">
                                            o_t =
                                            \sigma(W_o
                                            \cdot
                                            [h_{'{'}t-1{'}'},
                                            x_t] + b_o)
                                          </annotation>
                                        </semantics>
                                      </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.5806em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal">o</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>σ</span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>W</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '-0.1389em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">o</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">⋅</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mopen">[</span><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mbin mtight">−</span><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2083em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">]</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord"><span className="mord mathnormal">b</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">o</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">)</span></span></span></span></span>
                                <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                        <semantics>
                                          <mrow>
                                            <msub>
                                              <mi>h
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo>=</mo>
                                            <msub>
                                              <mi>o
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo>∗</mo>
                                            <mi>tanh
                                            </mi>
                                            <mo>⁡</mo>
                                            <mo stretchy="false">
                                              (</mo>
                                            <msub>
                                              <mi>C
                                              </mi>
                                              <mi>t
                                              </mi>
                                            </msub>
                                            <mo stretchy="false">
                                              )</mo>
                                          </mrow>
                                          <annotation encoding="application/x-tex">
                                            h_t = o_t *
                                            \tanh(C_t)
                                          </annotation>
                                        </semantics>
                                      </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8444em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '0.6153em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal">o</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">∗</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mop">tanh</span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.07153em'}}>C</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '-0.0715em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">)</span></span></span></span></span>
                              </p>
                            </li>
                          </ol>
                          <h4>Building an LSTM with Keras</h4>
                          <p>Keras simplifies the process of building and
                            training LSTM networks. Below, we
                            demonstrate how to construct an LSTM for
                            sequence prediction using the IMDB dataset
                            of movie reviews.</p>
                          <p><strong>Loading and Preprocessing the
                              Data</strong></p>
                          <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> tensorflow <span className="hljs-keyword">as</span> tf{"\n"}<span className="hljs-keyword">from</span> tensorflow.keras <span className="hljs-keyword">import</span> layers, models{"\n"}<span className="hljs-keyword">from</span> tensorflow.keras.datasets <span className="hljs-keyword">import</span> imdb{"\n"}<span className="hljs-keyword">from</span> tensorflow.keras.preprocessing <span className="hljs-keyword">import</span> sequence{"\n"}{"\n"}<span className="hljs-comment"># Load the dataset</span>{"\n"}max_features = <span className="hljs-number">20000</span>{"\n"}max_len = <span className="hljs-number">100</span>{"\n"}(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features){"\n"}{"\n"}<span className="hljs-comment"># Preprocess the data</span>{"\n"}X_train = sequence.pad_sequences(X_train, maxlen=max_len){"\n"}X_test = sequence.pad_sequences(X_test, maxlen=max_len){"\n"}</code></div></pre></div>
                        <p><strong>Building the LSTM</strong></p>
                        <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Build the LSTM</span>{"\n"}model = models.Sequential(){"\n"}model.add(layers.Embedding(max_features, <span className="hljs-number">128</span>, input_length=max_len)){"\n"}model.add(layers.LSTM(<span className="hljs-number">64</span>)){"\n"}model.add(layers.Dense(<span className="hljs-number">1</span>, activation=<span className="hljs-string">'sigmoid'</span>)){"\n"}{"\n"}<span className="hljs-comment"># Compile the model</span>{"\n"}model.<span className="hljs-built_in">compile</span>(optimizer=<span className="hljs-string">'adam'</span>, loss=<span className="hljs-string">'binary_crossentropy'</span>, metrics=[<span className="hljs-string">'accuracy'</span>]){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train, epochs=<span className="hljs-number">3</span>, batch_size=<span className="hljs-number">64</span>, validation_split=<span className="hljs-number">0.2</span>){"\n"}</code></div></pre></div>
                      <p><strong>Evaluating the LSTM</strong></p>
                      <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Evaluate the model</span>{"\n"}loss, accuracy = model.evaluate(X_test, y_test){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Loss: <span className="hljs-subst">{"{"}loss{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Accuracy: <span className="hljs-subst">{"{"}accuracy{"}"}</span>'</span>){"\n"}</code></div></pre></div>
                    <h4>Gated Recurrent Unit (GRU)</h4>
                    <p>GRUs are a simplified version of LSTMs that
                      combine the forget and input gates into a
                      single update gate. They are computationally
                      efficient and perform well on various
                      sequence tasks.</p>
                    <p><strong>Key Components of GRU:</strong></p>
                    <ol>
                      <li>
                        <p><strong>Update Gate</strong>: Decides
                          what information to update.
                          <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                  <semantics>
                                    <mrow>
                                      <msub>
                                        <mi>z
                                        </mi>
                                        <mi>t
                                        </mi>
                                      </msub>
                                      <mo>=</mo>
                                      <mi>σ</mi>
                                      <mo stretchy="false">
                                        (</mo>
                                      <msub>
                                        <mi>W
                                        </mi>
                                        <mi>z
                                        </mi>
                                      </msub>
                                      <mo>⋅</mo>
                                      <mo stretchy="false">
                                        [</mo>
                                      <msub>
                                        <mi>h
                                        </mi>
                                        <mrow>
                                          <mi>t
                                          </mi>
                                          <mo>−
                                          </mo>
                                          <mn>1
                                          </mn>
                                        </mrow>
                                      </msub>
                                      <mo separator="true">
                                        ,</mo>
                                      <msub>
                                        <mi>x
                                        </mi>
                                        <mi>t
                                        </mi>
                                      </msub>
                                      <mo stretchy="false">
                                        ]</mo>
                                      <mo>+</mo>
                                      <msub>
                                        <mi>b
                                        </mi>
                                        <mi>z
                                        </mi>
                                      </msub>
                                      <mo stretchy="false">
                                        )</mo>
                                    </mrow>
                                    <annotation encoding="application/x-tex">
                                      z_t =
                                      \sigma(W_z
                                      \cdot
                                      [h_{'{'}t-1{'}'},
                                      x_t] + b_z)
                                    </annotation>
                                  </semantics>
                                </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.5806em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.04398em'}}>z</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '-0.044em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>σ</span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>W</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '-0.1389em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.04398em'}}>z</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">⋅</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mopen">[</span><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mbin mtight">−</span><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2083em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">]</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord"><span className="mord mathnormal">b</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.04398em'}}>z</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">)</span></span></span></span></span>
                        </p>
                      </li>
                      <li>
                        <p><strong>Reset Gate</strong>: Decides
                          what information to discard from the
                          previous hidden state.
                          <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                  <semantics>
                                    <mrow>
                                      <msub>
                                        <mi>r
                                        </mi>
                                        <mi>t
                                        </mi>
                                      </msub>
                                      <mo>=</mo>
                                      <mi>σ</mi>
                                      <mo stretchy="false">
                                        (</mo>
                                      <msub>
                                        <mi>W
                                        </mi>
                                        <mi>r
                                        </mi>
                                      </msub>
                                      <mo>⋅</mo>
                                      <mo stretchy="false">
                                        [</mo>
                                      <msub>
                                        <mi>h
                                        </mi>
                                        <mrow>
                                          <mi>t
                                          </mi>
                                          <mo>−
                                          </mo>
                                          <mn>1
                                          </mn>
                                        </mrow>
                                      </msub>
                                      <mo separator="true">
                                        ,</mo>
                                      <msub>
                                        <mi>x
                                        </mi>
                                        <mi>t
                                        </mi>
                                      </msub>
                                      <mo stretchy="false">
                                        ]</mo>
                                      <mo>+</mo>
                                      <msub>
                                        <mi>b
                                        </mi>
                                        <mi>r
                                        </mi>
                                      </msub>
                                      <mo stretchy="false">
                                        )</mo>
                                    </mrow>
                                    <annotation encoding="application/x-tex">
                                      r_t =
                                      \sigma(W_r
                                      \cdot
                                      [h_{'{'}t-1{'}'},
                                      x_t] + b_r)
                                    </annotation>
                                  </semantics>
                                </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.5806em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.02778em'}}>r</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '-0.0278em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal" style={{marginRight: '0.03588em'}}>σ</span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>W</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '-0.1389em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.02778em'}}>r</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">⋅</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mopen">[</span><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mbin mtight">−</span><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2083em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">]</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord"><span className="mord mathnormal">b</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1514em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight" style={{marginRight: '0.02778em'}}>r</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">)</span></span></span></span></span>
                        </p>
                      </li>
                      <li>
                        <p><strong>Current Memory
                            Content</strong>: The new memory
                          content.
                          <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                  <semantics>
                                    <mrow>
                                      <msub>
                                        <mover accent="true">
                                          <mi>h
                                          </mi>
                                          <mo>~
                                          </mo>
                                        </mover>
                                        <mi>t
                                        </mi>
                                      </msub>
                                      <mo>=</mo>
                                      <mi>tanh
                                      </mi>
                                      <mo>⁡</mo>
                                      <mo stretchy="false">
                                        (</mo>
                                      <msub>
                                        <mi>W
                                        </mi>
                                        <mi>h
                                        </mi>
                                      </msub>
                                      <mo>⋅</mo>
                                      <mo stretchy="false">
                                        [</mo>
                                      <msub>
                                        <mi>r
                                        </mi>
                                        <mi>t
                                        </mi>
                                      </msub>
                                      <mo>∗</mo>
                                      <msub>
                                        <mi>h
                                        </mi>
                                        <mrow>
                                          <mi>t
                                          </mi>
                                          <mo>−
                                          </mo>
                                          <mn>1
                                          </mn>
                                        </mrow>
                                      </msub>
                                      <mo separator="true">
                                        ,</mo>
                                      <msub>
                                        <mi>x
                                        </mi>
                                        <mi>t
                                        </mi>
                                      </msub>
                                      <mo stretchy="false">
                                        ]</mo>
                                      <mo>+</mo>
                                      <msub>
                                        <mi>b
                                        </mi>
                                        <mi>h
                                        </mi>
                                      </msub>
                                      <mo stretchy="false">
                                        )</mo>
                                    </mrow>
                                    <annotation encoding="application/x-tex">
                                      \tilde{'{'}h{'}'}_t
                                      = \tanh(W_h
                                      \cdot [r_t *
                                      h_{'{'}t-1{'}'},
                                      x_t] + b_h)
                                    </annotation>
                                  </semantics>
                                </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1.0813em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord accent"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.9313em'}}><span style={{top: '-3em'}}><span className="pstrut" style={{height: '3em'}} /><span className="mord mathnormal">h</span></span><span style={{top: '-3.6134em'}}><span className="pstrut" style={{height: '3em'}} /><span className="accent-body" style={{left: '-0.25em'}}><span className="mord">~</span></span></span></span></span></span></span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mop">tanh</span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.13889em'}}>W</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3361em'}}><span style={{top: '-2.55em', marginLeft: '-0.1389em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">h</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">⋅</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mopen">[</span><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.02778em'}}>r</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '-0.0278em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">∗</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mbin mtight">−</span><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2083em'}}><span /></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal">x</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">]</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord"><span className="mord mathnormal">b</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3361em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">h</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">)</span></span></span></span></span>
                        </p>
                      </li>
                      <li>
                        <p><strong>Hidden State</strong>: The
                          final hidden state.
                          <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                                  <semantics>
                                    <mrow>
                                      <msub>
                                        <mi>h
                                        </mi>
                                        <mi>t
                                        </mi>
                                      </msub>
                                      <mo>=</mo>
                                      <mo stretchy="false">
                                        (</mo>
                                      <mn>1</mn>
                                      <mo>−</mo>
                                      <msub>
                                        <mi>z
                                        </mi>
                                        <mi>t
                                        </mi>
                                      </msub>
                                      <mo stretchy="false">
                                        )</mo>
                                      <mo>∗</mo>
                                      <msub>
                                        <mi>h
                                        </mi>
                                        <mrow>
                                          <mi>t
                                          </mi>
                                          <mo>−
                                          </mo>
                                          <mn>1
                                          </mn>
                                        </mrow>
                                      </msub>
                                      <mo>+</mo>
                                      <msub>
                                        <mi>z
                                        </mi>
                                        <mi>t
                                        </mi>
                                      </msub>
                                      <mo>∗</mo>
                                      <msub>
                                        <mover accent="true">
                                          <mi>h
                                          </mi>
                                          <mo>~
                                          </mo>
                                        </mover>
                                        <mi>t
                                        </mi>
                                      </msub>
                                    </mrow>
                                    <annotation encoding="application/x-tex">
                                      h_t = (1 -
                                      z_t) *
                                      h_{'{'}t-1{'}'} +
                                      z_t *
                                      \tilde{'{'}h{'}'}_t
                                    </annotation>
                                  </semantics>
                                </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.8444em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mopen">(</span><span className="mord">1</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">−</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.04398em'}}>z</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '-0.044em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mclose">)</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">∗</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.9028em', verticalAlign: '-0.2083em'}} /><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3011em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mbin mtight">−</span><span className="mord mtight">1</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2083em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '0.6153em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.04398em'}}>z</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '-0.044em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">∗</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1.0813em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord accent"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.9313em'}}><span style={{top: '-3em'}}><span className="pstrut" style={{height: '3em'}} /><span className="mord mathnormal">h</span></span><span style={{top: '-3.6134em'}}><span className="pstrut" style={{height: '3em'}} /><span className="accent-body" style={{left: '-0.25em'}}><span className="mord">~</span></span></span></span></span></span></span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
                        </p>
                      </li>
                    </ol>
                    <p><strong>Building a GRU with Keras</strong>
                    </p>
                    <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Build the GRU</span>{"\n"}model = models.Sequential(){"\n"}model.add(layers.Embedding(max_features, <span className="hljs-number">128</span>, input_length=max_len)){"\n"}model.add(layers.GRU(<span className="hljs-number">64</span>)){"\n"}model.add(layers.Dense(<span className="hljs-number">1</span>, activation=<span className="hljs-string">'sigmoid'</span>)){"\n"}{"\n"}<span className="hljs-comment"># Compile the model</span>{"\n"}model.<span className="hljs-built_in">compile</span>(optimizer=<span className="hljs-string">'adam'</span>, loss=<span className="hljs-string">'binary_crossentropy'</span>, metrics=[<span className="hljs-string">'accuracy'</span>]){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train, epochs=<span className="hljs-number">3</span>, batch_size=<span className="hljs-number">64</span>, validation_split=<span className="hljs-number">0.2</span>){"\n"}</code></div></pre></div>
                  <p><strong>Evaluating the GRU</strong></p>
                  <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Evaluate the model</span>{"\n"}loss, accuracy = model.evaluate(X_test, y_test){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Loss: <span className="hljs-subst">{"{"}loss{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Accuracy: <span className="hljs-subst">{"{"}accuracy{"}"}</span>'</span>){"\n"}</code></div></pre></div>
                <h4>Bidirectional RNN</h4>
                <p>Bidirectional RNNs process the input sequence
                  in both forward and backward directions,
                  capturing dependencies from both past and
                  future states. This is particularly useful
                  for tasks like language modeling and speech
                  recognition.</p>
                <p><strong>Building a Bidirectional LSTM with
                    Keras</strong></p>
                <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Build the Bidirectional LSTM</span>{"\n"}model = models.Sequential(){"\n"}model.add(layers.Embedding(max_features, <span className="hljs-number">128</span>, input_length=max_len)){"\n"}model.add(layers.Bidirectional(layers.LSTM(<span className="hljs-number">64</span>))){"\n"}model.add(layers.Dense(<span className="hljs-number">1</span>, activation=<span className="hljs-string">'sigmoid'</span>)){"\n"}{"\n"}<span className="hljs-comment"># Compile the model</span>{"\n"}model.<span className="hljs-built_in">compile</span>(optimizer=<span className="hljs-string">'adam'</span>, loss=<span className="hljs-string">'binary_crossentropy'</span>, metrics=[<span className="hljs-string">'accuracy'</span>]){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit(X_train, y_train, epochs=<span className="hljs-number">3</span>, batch_size=<span className="hljs-number">64</span>, validation_split=<span className="hljs-number">0.2</span>){"\n"}</code></div></pre></div>
              <p><strong>Evaluating the Bidirectional
                  LSTM</strong></p>
              <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Evaluate the model</span>{"\n"}loss, accuracy = model.evaluate(X_test, y_test){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Loss: <span className="hljs-subst">{"{"}loss{"}"}</span>'</span>){"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Accuracy: <span className="hljs-subst">{"{"}accuracy{"}"}</span>'</span>){"\n"}</code></div></pre></div>
            <h4>Sequence-to-Sequence (Seq2Seq) Models</h4>
            <p>Seq2Seq models are used for tasks that
              require generating a sequence from another
              sequence, such as machine translation and
              text summarization. They consist of an
              encoder and a decoder.</p>
            <p><strong>Key Components of Seq2Seq:</strong>
            </p>
            <ol>
              <li><strong>Encoder</strong>: Processes the
                input sequence and produces a context
                vector.</li>
              <li><strong>Decoder</strong>: Generates the
                output sequence based on the context
                vector.</li>
            </ol>
            <p><strong>Building a Seq2Seq Model with
                Keras</strong></p>
            <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Define the encoder</span>{"\n"}encoder_inputs = layers.Input(shape=(<span className="hljs-literal">None</span>, num_encoder_tokens)){"\n"}encoder = layers.LSTM(latent_dim, return_state=<span className="hljs-literal">True</span>){"\n"}encoder_outputs, state_h, state_c = encoder(encoder_inputs){"\n"}encoder_states = [state_h, state_c]{"\n"}{"\n"}<span className="hljs-comment"># Define the decoder</span>{"\n"}decoder_inputs = layers.Input(shape=(<span className="hljs-literal">None</span>, num_decoder_tokens)){"\n"}decoder_lstm = layers.LSTM(latent_dim, return_sequences=<span className="hljs-literal">True</span>, return_state=<span className="hljs-literal">True</span>){"\n"}decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states){"\n"}decoder_dense = layers.Dense(num_decoder_tokens, activation=<span className="hljs-string">'softmax'</span>){"\n"}decoder_outputs = decoder_dense(decoder_outputs){"\n"}{"\n"}<span className="hljs-comment"># Define the model</span>{"\n"}model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs){"\n"}{"\n"}<span className="hljs-comment"># Compile the model</span>{"\n"}model.<span className="hljs-built_in">compile</span>(optimizer=<span className="hljs-string">'adam'</span>, loss=<span className="hljs-string">'categorical_crossentropy'</span>, metrics=[<span className="hljs-string">'accuracy'</span>]){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=<span className="hljs-number">100</span>, batch_size=<span className="hljs-number">64</span>, validation_split=<span className="hljs-number">0.2</span>){"\n"}</code></div></pre></div>
          <h4>Attention Mechanism</h4>
          <p>The attention mechanism allows the model to
            focus on specific parts of the input
            sequence when generating each output token,
            improving the performance of Seq2Seq models.
          </p>
          <p><strong>Key Components of Attention:</strong>
          </p>
          <ol>
            <li>
              <p><strong>Attention Weights</strong>:
                Weights that determine the
                importance of each input token for
                generating the current output token.
                <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                        <semantics>
                          <mrow>
                            <msub>
                              <mi>α
                              </mi>
                              <mrow>
                                <mi>t
                                </mi>
                                <mo separator="true">
                                  ,
                                </mo>
                                <msup>
                                  <mi>t
                                  </mi>
                                  <mo mathvariant="normal" lspace="0em" rspace="0em">
                                    ′
                                  </mo>
                                </msup>
                              </mrow>
                            </msub>
                            <mo>=</mo>
                            <mfrac>
                              <mrow>
                                <mi>exp
                                </mi>
                                <mo>⁡
                                </mo>
                                <mo stretchy="false">
                                  (
                                </mo>
                                <msub>
                                  <mi>e
                                  </mi>
                                  <mrow>
                                    <mi>t
                                    </mi>
                                    <mo separator="true">
                                      ,
                                    </mo>
                                    <msup>
                                      <mi>t
                                      </mi>
                                      <mo mathvariant="normal" lspace="0em" rspace="0em">
                                        ′
                                      </mo>
                                    </msup>
                                  </mrow>
                                </msub>
                                <mo stretchy="false">
                                  )
                                </mo>
                              </mrow>
                              <mrow>
                                <msub>
                                  <mo>∑
                                  </mo>
                                  <msup>
                                    <mi>t
                                    </mi>
                                    <mo mathvariant="normal" lspace="0em" rspace="0em">
                                      ′
                                    </mo>
                                  </msup>
                                </msub>
                                <mi>exp
                                </mi>
                                <mo>⁡
                                </mo>
                                <mo stretchy="false">
                                  (
                                </mo>
                                <msub>
                                  <mi>e
                                  </mi>
                                  <mrow>
                                    <mi>t
                                    </mi>
                                    <mo separator="true">
                                      ,
                                    </mo>
                                    <msup>
                                      <mi>t
                                      </mi>
                                      <mo mathvariant="normal" lspace="0em" rspace="0em">
                                        ′
                                      </mo>
                                    </msup>
                                  </mrow>
                                </msub>
                                <mo stretchy="false">
                                  )
                                </mo>
                              </mrow>
                            </mfrac>
                          </mrow>
                          <annotation encoding="application/x-tex">
                            \alpha_{'{'}t,
                            t'{'}'} =
                            \frac{'{'}\exp(e_{'{'}t,
                            t'{'}'}){'}'}{'{'}\sum_{'{'}t'{'}'}
                            \exp(e_{'{'}t,
                            t'{'}'}){'}'}
                          </annotation>
                        </semantics>
                      </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.7167em', verticalAlign: '-0.2861em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.0037em'}}>α</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.328em'}}><span style={{top: '-2.55em', marginLeft: '-0.0037em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mpunct mtight">,</span><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.6828em'}}><span style={{top: '-2.786em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mtight">′</span></span></span></span></span></span></span></span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2861em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.7413em', verticalAlign: '-0.6256em'}} /><span className="mord"><span className="mopen nulldelimiter" /><span className="mfrac"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '1.1156em'}}><span style={{top: '-2.655em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mop mtight"><span className="mop op-symbol small-op mtight" style={{position: 'relative', top: '0em'}}>∑</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2854em'}}><span style={{top: '-2.2854em', marginLeft: '0em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.6068em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8496em'}}><span style={{top: '-2.8496em', marginRight: '0.1em'}}><span className="pstrut" style={{height: '2.5556em'}} /><span className="mord mtight"><span className="mord mtight">′</span></span></span></span></span></span></span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.3214em'}}><span /></span></span></span></span></span><span className="mspace mtight" style={{marginRight: '0.1952em'}} /><span className="mop mtight"><span className="mtight">e</span><span className="mtight">x</span><span className="mtight">p</span></span><span className="mopen mtight">(</span><span className="mord mtight"><span className="mord mathnormal mtight">e</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3448em'}}><span style={{top: '-2.3448em', marginLeft: '0em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.6068em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mpunct mtight">,</span><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8496em'}}><span style={{top: '-2.8496em', marginRight: '0.1em'}}><span className="pstrut" style={{height: '2.5556em'}} /><span className="mord mtight"><span className="mord mtight">′</span></span></span></span></span></span></span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4009em'}}><span /></span></span></span></span></span><span className="mclose mtight">)</span></span></span></span><span style={{top: '-3.23em'}}><span className="pstrut" style={{height: '3em'}} /><span className="frac-line" style={{borderBottomWidth: '0.04em'}} /></span><span style={{top: '-3.5906em'}}><span className="pstrut" style={{height: '3em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mop mtight"><span className="mtight">e</span><span className="mtight">x</span><span className="mtight">p</span></span><span className="mopen mtight">(</span><span className="mord mtight"><span className="mord mathnormal mtight">e</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.3448em'}}><span style={{top: '-2.3448em', marginLeft: '0em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.6068em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mpunct mtight">,</span><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.8496em'}}><span style={{top: '-2.8496em', marginRight: '0.1em'}}><span className="pstrut" style={{height: '2.5556em'}} /><span className="mord mtight"><span className="mord mtight">′</span></span></span></span></span></span></span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.4009em'}}><span /></span></span></span></span></span><span className="mclose mtight">)</span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.6256em'}}><span /></span></span></span></span><span className="mclose nulldelimiter" /></span></span></span></span></span>
              </p>
            </li>
            <li>
              <p><strong>Context Vector</strong>: A
                weighted sum of the input tokens
                based on the attention weights.
                <span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                        <semantics>
                          <mrow>
                            <msub>
                              <mi>c
                              </mi>
                              <mi>t
                              </mi>
                            </msub>
                            <mo>=</mo>
                            <msub>
                              <mo>∑
                              </mo>
                              <msup>
                                <mi>t
                                </mi>
                                <mo mathvariant="normal" lspace="0em" rspace="0em">
                                  ′
                                </mo>
                              </msup>
                            </msub>
                            <msub>
                              <mi>α
                              </mi>
                              <mrow>
                                <mi>t
                                </mi>
                                <mo separator="true">
                                  ,
                                </mo>
                                <msup>
                                  <mi>t
                                  </mi>
                                  <mo mathvariant="normal" lspace="0em" rspace="0em">
                                    ′
                                  </mo>
                                </msup>
                              </mrow>
                            </msub>
                            <msub>
                              <mi>h
                              </mi>
                              <msup>
                                <mi>t
                                </mi>
                                <mo mathvariant="normal" lspace="0em" rspace="0em">
                                  ′
                                </mo>
                              </msup>
                            </msub>
                          </mrow>
                          <annotation encoding="application/x-tex">
                            c_t =
                            \sum_{'{'}t'{'}'}
                            \alpha_{'{'}t,
                            t'{'}'} h_{'{'}t'{'}'}
                          </annotation>
                        </semantics>
                      </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.5806em', verticalAlign: '-0.15em'}} /><span className="mord"><span className="mord mathnormal">c</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.2806em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mathnormal mtight">t</span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">=</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1.0497em', verticalAlign: '-0.2997em'}} /><span className="mop"><span className="mop op-symbol small-op" style={{position: 'relative', top: '0em'}}>∑</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.1783em'}}><span style={{top: '-2.4003em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.6828em'}}><span style={{top: '-2.786em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mtight">′</span></span></span></span></span></span></span></span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2997em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal" style={{marginRight: '0.0037em'}}>α</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.328em'}}><span style={{top: '-2.55em', marginLeft: '-0.0037em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="mpunct mtight">,</span><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.6828em'}}><span style={{top: '-2.786em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mtight">′</span></span></span></span></span></span></span></span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.2861em'}}><span /></span></span></span></span></span><span className="mord"><span className="mord mathnormal">h</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.328em'}}><span style={{top: '-2.55em', marginLeft: '0em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight"><span className="mord mathnormal mtight">t</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.6828em'}}><span style={{top: '-2.786em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mtight">′</span></span></span></span></span></span></span></span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span></span></span></span></span>
              </p>
            </li>
          </ol>
          <p><strong>Building a Seq2Seq Model with
              Attention</strong></p>
          <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-comment"># Define the encoder</span>{"\n"}encoder_inputs = layers.Input(shape=(<span className="hljs-literal">None</span>, num_encoder_tokens)){"\n"}encoder = layers.LSTM(latent_dim, return_state=<span className="hljs-literal">True</span>){"\n"}encoder_outputs, state_h, state_c = encoder(encoder_inputs){"\n"}encoder_states = [state_h, state_c]{"\n"}{"\n"}<span className="hljs-comment"># Define the decoder with attention</span>{"\n"}decoder_inputs = layers.Input(shape=(<span className="hljs-literal">None</span>, num_decoder_tokens)){"\n"}decoder_lstm = layers.LSTM(latent_dim, return_sequences=<span className="hljs-literal">True</span>, return_state=<span className="hljs-literal">True</span>){"\n"}decoder_lstm_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states){"\n"}attention = layers.AdditiveAttention(){"\n"}context_vector = attention([decoder_lstm_outputs, encoder_outputs]){"\n"}decoder_concat_input = layers.Concatenate(axis=-<span className="hljs-number">1</span>)([context_vector, decoder_lstm_outputs]){"\n"}decoder_dense = layers.Dense(num_decoder_tokens, activation=<span className="hljs-string">'softmax'</span>){"\n"}decoder_outputs = decoder_dense(decoder_concat_input){"\n"}{"\n"}<span className="hljs-comment"># Define the model</span>{"\n"}model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs){"\n"}{"\n"}<span className="hljs-comment"># Compile the model</span>{"\n"}model.<span className="hljs-built_in">compile</span>(optimizer=<span className="hljs-string">'adam'</span>, loss=<span className="hljs-string">'categorical_crossentropy'</span>, metrics=[<span className="hljs-string">'accuracy'</span>]){"\n"}{"\n"}<span className="hljs-comment"># Train the model</span>{"\n"}model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=<span className="hljs-number">100</span>, batch_size=<span className="hljs-number">64</span>, validation_split=<span className="hljs-number">0.2</span>){"\n"}</code></div></pre></div>
        <h4>Practical Tips for Building RNNs</h4>
        <ol>
          <li><strong>Choose the Right RNN
              Type</strong>: Select the
            appropriate RNN type (e.g., LSTM, GRU)
            based on the complexity and nature of
            your task.</li>
          <li><strong>Use Regularization</strong>:
            Apply techniques like dropout and
            recurrent dropout to prevent
            overfitting.</li>
          <li><strong>Monitor and Adjust Learning
              Rate</strong>: Use learning rate
            scheduling and early stopping to improve
            training stability.</li>
          <li><strong>Handle Long Sequences</strong>:
            For long sequences, consider using
            truncated backpropagation through time
            (TBPTT) to reduce computational
            complexity.</li>
          <li><strong>Use Pre-trained
              Embeddings</strong>: Leverage
            pre-trained word embeddings (e.g.,
            GloVe, Word2Vec) for natural language
            processing tasks to improve performance.
          </li>
        </ol>
        <h4>Conclusion</h4>
        <p>Recurrent Neural Networks (RNNs) are powerful
          models for sequential data, enabling the
          processing of time series, natural language,
          and other temporally dependent data. This
          chapter covered the fundamentals of RNNs,
          including LSTMs, GRUs, Bidirectional RNNs,
          Seq2Seq models, and the attention mechanism.
          By understanding and applying these
          techniques, you can build robust RNNs that
          achieve high performance on a wide range of
          sequential tasks.</p>
      </div>

    )
}

export default ChapterTwelve;