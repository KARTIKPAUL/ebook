import React from "react";

function ChapterFifteenth(){
    return(
        <div className="w-full text-token-text-primary" dir="auto" id="ch-15" data-testid="conversation-turn-33" data-scroll-anchor="false">
    <div className="py-2 juice:py-[18px] px-3 text-base md:px-4 m-auto">
      <div className="mx-auto flex flex-1 gap-3 text-base juice:gap-4 juice:md:gap-6 md:max-w-3xl lg:max-w-[40rem] xl:max-w-[48rem]">
        <div className="group/conversation-turn relative flex w-full min-w-0 flex-col agent-turn">
          <div className="flex-col gap-1 md:gap-3">
            <div className="flex flex-grow flex-col max-w-full">
              <div data-message-author-role="assistant" data-message-id="4300f02c-205a-40ad-b9a0-fce4728acbb5" dir="auto" className="text-message flex flex-col items-start whitespace-pre-wrap break-words [.text-message+&]:mt-5 juice:w-full juice:items-end overflow-x-auto gap-2">
                <div className="flex w-full flex-col gap-1 juice:empty:hidden juice:first:pt-[3px]">
                  <div className="markdown prose w-full break-words dark:prose-invert dark">
                    <h3>Chapter 15: Reinforcement Learning</h3>
                    <hr />
                    <p>Reinforcement learning (RL) is a type of
                      machine learning where an agent learns to
                      make decisions by interacting with an
                      environment. The agent receives rewards or
                      penalties based on its actions, and its goal
                      is to maximize the cumulative reward over
                      time. This chapter explores the fundamentals
                      of reinforcement learning, key concepts, and
                      how to implement RL algorithms using Python
                      and popular libraries like OpenAI Gym and
                      TensorFlow.</p>
                    <h4>Introduction to Reinforcement Learning</h4>
                    <p>Reinforcement learning is inspired by
                      behavioral psychology and involves an agent,
                      environment, actions, states, and rewards.
                      Unlike supervised learning, where the model
                      is trained on a fixed dataset, RL involves
                      learning from the consequences of actions
                      through trial and error.</p>
                    <p><strong>Key Concepts:</strong></p>
                    <ul>
                      <li><strong>Agent</strong>: The learner or
                        decision-maker.</li>
                      <li><strong>Environment</strong>: The world
                        with which the agent interacts.</li>
                      <li><strong>State</strong>: A representation
                        of the environment at a given time.</li>
                      <li><strong>Action</strong>: A decision made
                        by the agent that affects the state.
                      </li>
                      <li><strong>Reward</strong>: A scalar
                        feedback signal indicating the quality
                        of the agent's action.</li>
                      <li><strong>Policy</strong>: A strategy used
                        by the agent to determine actions based
                        on states.</li>
                      <li><strong>Value Function</strong>: A
                        function that estimates the expected
                        cumulative reward from a given state or
                        state-action pair.</li>
                    </ul>
                    <h4>Reinforcement Learning Framework</h4>
                    <p>The RL framework can be formalized as a
                      Markov Decision Process (MDP), defined by
                      the following components:</p>
                    <ol>
                      <li><strong>State Space (S)</strong>: The
                        set of all possible states.</li>
                      <li><strong>Action Space (A)</strong>: The
                        set of all possible actions.</li>
                      <li><strong>Transition Function
                          (T)</strong>: Describes the
                        probability of moving from one state to
                        another given an action.</li>
                      <li><strong>Reward Function (R)</strong>:
                        Maps state-action pairs to rewards.</li>
                      <li><strong>Policy (π)</strong>: Defines the
                        agent's behavior, mapping states to
                        actions.</li>
                    </ol>
                    <h4>Types of Reinforcement Learning</h4>
                    <p>There are two main types of reinforcement
                      learning:</p>
                    <ol>
                      <li><strong>Model-Based RL</strong>: The
                        agent builds a model of the environment
                        and uses it to make decisions.</li>
                      <li><strong>Model-Free RL</strong>: The
                        agent learns directly from interactions
                        with the environment without building a
                        model.</li>
                    </ol>
                    <p><strong>Example: Model-Free RL</strong></p>
                    <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> gym{"\n"}{"\n"}<span className="hljs-comment"># Create the environment</span>{"\n"}env = gym.make(<span className="hljs-string">'CartPole-v1'</span>){"\n"}{"\n"}<span className="hljs-comment"># Initialize the environment</span>{"\n"}state = env.reset(){"\n"}{"\n"}<span className="hljs-comment"># Define the policy (random actions for simplicity)</span>{"\n"}<span className="hljs-keyword">for</span> _ <span className="hljs-keyword">in</span> <span className="hljs-built_in">range</span>(<span className="hljs-number">1000</span>):{"\n"}{"    "}env.render(){"\n"}{"    "}action = env.action_space.sample(){"  "}<span className="hljs-comment"># Take a random action</span>{"\n"}{"    "}state, reward, done, info = env.step(action){"\n"}{"    "}<span className="hljs-keyword">if</span> done:{"\n"}{"        "}state = env.reset(){"\n"}{"\n"}env.close(){"\n"}</code></div></pre></div>
                  <h4>Q-Learning</h4>
                  <p>Q-Learning is a model-free RL algorithm that
                    learns the value of state-action pairs
                    (Q-values) to determine the best policy. The
                    Q-value is updated using the Bellman
                    equation:</p>
                  <p><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                            <semantics>
                              <mrow>
                                <mi>Q</mi>
                                <mo stretchy="false">
                                  (</mo>
                                <mi>s</mi>
                                <mo separator="true">
                                  ,</mo>
                                <mi>a</mi>
                                <mo stretchy="false">
                                  )</mo>
                                <mo>←</mo>
                                <mi>Q</mi>
                                <mo stretchy="false">
                                  (</mo>
                                <mi>s</mi>
                                <mo separator="true">
                                  ,</mo>
                                <mi>a</mi>
                                <mo stretchy="false">
                                  )</mo>
                                <mo>+</mo>
                                <mi>α</mi>
                                <mrow>
                                  <mo fence="true">
                                    [</mo>
                                  <mi>r</mi>
                                  <mo>+</mo>
                                  <mi>γ</mi>
                                  <msub>
                                    <mrow>
                                      <mi>max
                                      </mi>
                                      <mo>⁡
                                      </mo>
                                    </mrow>
                                    <msup>
                                      <mi>a
                                      </mi>
                                      <mo mathvariant="normal" lspace="0em" rspace="0em">
                                        ′
                                      </mo>
                                    </msup>
                                  </msub>
                                  <mi>Q</mi>
                                  <mo stretchy="false">
                                    (</mo>
                                  <msup>
                                    <mi>s</mi>
                                    <mo mathvariant="normal" lspace="0em" rspace="0em">
                                      ′</mo>
                                  </msup>
                                  <mo separator="true">
                                    ,</mo>
                                  <msup>
                                    <mi>a</mi>
                                    <mo mathvariant="normal" lspace="0em" rspace="0em">
                                      ′</mo>
                                  </msup>
                                  <mo stretchy="false">
                                    )</mo>
                                  <mo>−</mo>
                                  <mi>Q</mi>
                                  <mo stretchy="false">
                                    (</mo>
                                  <mi>s</mi>
                                  <mo separator="true">
                                    ,</mo>
                                  <mi>a</mi>
                                  <mo stretchy="false">
                                    )</mo>
                                  <mo fence="true">
                                    ]</mo>
                                </mrow>
                              </mrow>
                              <annotation encoding="application/x-tex">
                                Q(s, a) \leftarrow
                                Q(s, a) + \alpha
                                \left[ r + \gamma
                                \max_{'{'}a'{'}'} Q(s', a')
                                - Q(s, a) \right]
                              </annotation>
                            </semantics>
                          </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal">Q</span><span className="mopen">(</span><span className="mord mathnormal">s</span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord mathnormal">a</span><span className="mclose">)</span><span className="mspace" style={{marginRight: '0.2778em'}} /><span className="mrel">←</span><span className="mspace" style={{marginRight: '0.2778em'}} /></span><span className="base"><span className="strut" style={{height: '1em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal">Q</span><span className="mopen">(</span><span className="mord mathnormal">s</span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord mathnormal">a</span><span className="mclose">)</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /></span><span className="base"><span className="strut" style={{height: '1.0019em', verticalAlign: '-0.25em'}} /><span className="mord mathnormal" style={{marginRight: '0.0037em'}}>α</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="minner"><span className="mopen delimcenter" style={{top: '0em'}}>[</span><span className="mord mathnormal" style={{marginRight: '0.02778em'}}>r</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">+</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mord mathnormal" style={{marginRight: '0.05556em'}}>γ</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mop"><span className="mop">max</span><span className="msupsub"><span className="vlist-t vlist-t2"><span className="vlist-r"><span className="vlist" style={{height: '0.328em'}}><span style={{top: '-2.55em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight"><span className="mord mathnormal mtight">a</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.6828em'}}><span style={{top: '-2.786em', marginRight: '0.0714em'}}><span className="pstrut" style={{height: '2.5em'}} /><span className="sizing reset-size3 size1 mtight"><span className="mord mtight"><span className="mord mtight">′</span></span></span></span></span></span></span></span></span></span></span></span></span><span className="vlist-s">​</span></span><span className="vlist-r"><span className="vlist" style={{height: '0.15em'}}><span /></span></span></span></span></span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord mathnormal">Q</span><span className="mopen">(</span><span className="mord"><span className="mord mathnormal">s</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.7519em'}}><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight">′</span></span></span></span></span></span></span></span></span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord"><span className="mord mathnormal">a</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.7519em'}}><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight">′</span></span></span></span></span></span></span></span></span><span className="mclose">)</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mbin">−</span><span className="mspace" style={{marginRight: '0.2222em'}} /><span className="mord mathnormal">Q</span><span className="mopen">(</span><span className="mord mathnormal">s</span><span className="mpunct">,</span><span className="mspace" style={{marginRight: '0.1667em'}} /><span className="mord mathnormal">a</span><span className="mclose">)</span><span className="mclose delimcenter" style={{top: '0em'}}>]</span></span></span></span></span></span>
                  </p>
                  <p>Where:</p>
                  <ul>
                    <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                              <semantics>
                                <mrow>
                                  <mi>α</mi>
                                </mrow>
                                <annotation encoding="application/x-tex">
                                  \alpha
                                </annotation>
                              </semantics>
                            </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.4306em'}} /><span className="mord mathnormal" style={{marginRight: '0.0037em'}}>α</span></span></span></span></span>
                      is the learning rate.</li>
                    <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                              <semantics>
                                <mrow>
                                  <mi>γ</mi>
                                </mrow>
                                <annotation encoding="application/x-tex">
                                  \gamma
                                </annotation>
                              </semantics>
                            </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.625em', verticalAlign: '-0.1944em'}} /><span className="mord mathnormal" style={{marginRight: '0.05556em'}}>γ</span></span></span></span></span>
                      is the discount factor.</li>
                    <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                              <semantics>
                                <mrow>
                                  <mi>r</mi>
                                </mrow>
                                <annotation encoding="application/x-tex">
                                  r</annotation>
                              </semantics>
                            </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.4306em'}} /><span className="mord mathnormal" style={{marginRight: '0.02778em'}}>r</span></span></span></span></span>
                      is the reward.</li>
                    <li><span className="math math-inline"><span className="katex"><span className="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML">
                              <semantics>
                                <mrow>
                                  <msup>
                                    <mi>s</mi>
                                    <mo mathvariant="normal" lspace="0em" rspace="0em">
                                      ′</mo>
                                  </msup>
                                </mrow>
                                <annotation encoding="application/x-tex">
                                  s'</annotation>
                              </semantics>
                            </math></span><span className="katex-html" aria-hidden="true"><span className="base"><span className="strut" style={{height: '0.7519em'}} /><span className="mord"><span className="mord mathnormal">s</span><span className="msupsub"><span className="vlist-t"><span className="vlist-r"><span className="vlist" style={{height: '0.7519em'}}><span style={{top: '-3.063em', marginRight: '0.05em'}}><span className="pstrut" style={{height: '2.7em'}} /><span className="sizing reset-size6 size3 mtight"><span className="mord mtight"><span className="mord mtight">′</span></span></span></span></span></span></span></span></span></span></span></span></span>
                      is the next state.</li>
                  </ul>
                  <p><strong>Example: Q-Learning</strong></p>
                  <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> numpy <span className="hljs-keyword">as</span> np{"\n"}<span className="hljs-keyword">import</span> random{"\n"}<span className="hljs-keyword">import</span> gym{"\n"}{"\n"}<span className="hljs-comment"># Create the environment</span>{"\n"}env = gym.make(<span className="hljs-string">'FrozenLake-v0'</span>){"\n"}{"\n"}<span className="hljs-comment"># Initialize Q-table</span>{"\n"}q_table = np.zeros([env.observation_space.n, env.action_space.n]){"\n"}{"\n"}<span className="hljs-comment"># Define hyperparameters</span>{"\n"}alpha = <span className="hljs-number">0.1</span>{"\n"}gamma = <span className="hljs-number">0.99</span>{"\n"}epsilon = <span className="hljs-number">0.1</span>{"\n"}episodes = <span className="hljs-number">1000</span>{"\n"}{"\n"}<span className="hljs-comment"># Training the agent</span>{"\n"}<span className="hljs-keyword">for</span> episode <span className="hljs-keyword">in</span> <span className="hljs-built_in">range</span>(episodes):{"\n"}{"    "}state = env.reset(){"\n"}{"    "}done = <span className="hljs-literal">False</span>{"\n"}{"\n"}{"    "}<span className="hljs-keyword">while</span> <span className="hljs-keyword">not</span> done:{"\n"}{"        "}<span className="hljs-keyword">if</span> random.uniform(<span className="hljs-number">0</span>, <span className="hljs-number">1</span>) &lt; epsilon:{"\n"}{"            "}action = env.action_space.sample(){"  "}<span className="hljs-comment"># Explore action space</span>{"\n"}{"        "}<span className="hljs-keyword">else</span>:{"\n"}{"            "}action = np.argmax(q_table[state]){"  "}<span className="hljs-comment"># Exploit learned values</span>{"\n"}{"\n"}{"        "}next_state, reward, done, info = env.step(action){"\n"}{"\n"}{"        "}<span className="hljs-comment"># Update Q-value</span>{"\n"}{"        "}old_value = q_table[state, action]{"\n"}{"        "}next_max = np.<span className="hljs-built_in">max</span>(q_table[next_state]){"\n"}{"        "}new_value = (<span className="hljs-number">1</span> - alpha) * old_value + alpha * (reward + gamma * next_max){"\n"}{"        "}q_table[state, action] = new_value{"\n"}{"\n"}{"        "}state = next_state{"\n"}{"\n"}<span className="hljs-comment"># Evaluate the agent</span>{"\n"}total_rewards = <span className="hljs-number">0</span>{"\n"}<span className="hljs-keyword">for</span> _ <span className="hljs-keyword">in</span> <span className="hljs-built_in">range</span>(<span className="hljs-number">100</span>):{"\n"}{"    "}state = env.reset(){"\n"}{"    "}done = <span className="hljs-literal">False</span>{"\n"}{"    "}<span className="hljs-keyword">while</span> <span className="hljs-keyword">not</span> done:{"\n"}{"        "}action = np.argmax(q_table[state]){"\n"}{"        "}state, reward, done, info = env.step(action){"\n"}{"        "}total_rewards += reward{"\n"}{"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Average Reward: <span className="hljs-subst">{"{"}total_rewards / <span className="hljs-number">100</span>{"}"}</span>'</span>){"\n"}</code></div></pre></div>
                <h4>Deep Q-Networks (DQNs)</h4>
                <p>Deep Q-Networks combine Q-Learning with deep
                  neural networks to handle high-dimensional
                  state spaces. A DQN approximates the Q-value
                  function using a neural network.</p>
                <p><strong>Example: DQN with TensorFlow</strong>
                </p>
                <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> tensorflow <span className="hljs-keyword">as</span> tf{"\n"}<span className="hljs-keyword">from</span> tensorflow.keras <span className="hljs-keyword">import</span> layers{"\n"}<span className="hljs-keyword">import</span> numpy <span className="hljs-keyword">as</span> np{"\n"}<span className="hljs-keyword">import</span> gym{"\n"}{"\n"}<span className="hljs-comment"># Create the environment</span>{"\n"}env = gym.make(<span className="hljs-string">'CartPole-v1'</span>){"\n"}{"\n"}<span className="hljs-comment"># Define the Q-network</span>{"\n"}<span className="hljs-keyword">def</span> <span className="hljs-title function_">build_model</span>(<span className="hljs-params">state_shape, action_shape</span>):{"\n"}{"    "}model = tf.keras.Sequential([{"\n"}{"        "}layers.Dense(<span className="hljs-number">24</span>, activation=<span className="hljs-string">'relu'</span>, input_shape=state_shape),{"\n"}{"        "}layers.Dense(<span className="hljs-number">24</span>, activation=<span className="hljs-string">'relu'</span>),{"\n"}{"        "}layers.Dense(action_shape, activation=<span className="hljs-string">'linear'</span>){"\n"}{"    "}]){"\n"}{"    "}model.<span className="hljs-built_in">compile</span>(optimizer=tf.keras.optimizers.Adam(learning_rate=<span className="hljs-number">0.001</span>), loss=<span className="hljs-string">'mse'</span>){"\n"}{"    "}<span className="hljs-keyword">return</span> model{"\n"}{"\n"}<span className="hljs-comment"># Hyperparameters</span>{"\n"}gamma = <span className="hljs-number">0.99</span>{"\n"}epsilon = <span className="hljs-number">1.0</span>{"\n"}epsilon_min = <span className="hljs-number">0.01</span>{"\n"}epsilon_decay = <span className="hljs-number">0.995</span>{"\n"}batch_size = <span className="hljs-number">32</span>{"\n"}memory = []{"\n"}max_memory = <span className="hljs-number">2000</span>{"\n"}episodes = <span className="hljs-number">1000</span>{"\n"}{"\n"}<span className="hljs-comment"># Initialize model</span>{"\n"}state_shape = (env.observation_space.shape[<span className="hljs-number">0</span>],){"\n"}action_shape = env.action_space.n{"\n"}model = build_model(state_shape, action_shape){"\n"}{"\n"}<span className="hljs-comment"># Training the DQN</span>{"\n"}<span className="hljs-keyword">for</span> episode <span className="hljs-keyword">in</span> <span className="hljs-built_in">range</span>(episodes):{"\n"}{"    "}state = env.reset(){"\n"}{"    "}state = np.reshape(state, [<span className="hljs-number">1</span>, state_shape[<span className="hljs-number">0</span>]]){"\n"}{"    "}total_reward = <span className="hljs-number">0</span>{"\n"}{"\n"}{"    "}<span className="hljs-keyword">for</span> step <span className="hljs-keyword">in</span> <span className="hljs-built_in">range</span>(<span className="hljs-number">200</span>):{"\n"}{"        "}<span className="hljs-keyword">if</span> np.random.rand() &lt;= epsilon:{"\n"}{"            "}action = np.random.randint(action_shape){"\n"}{"        "}<span className="hljs-keyword">else</span>:{"\n"}{"            "}q_values = model.predict(state){"\n"}{"            "}action = np.argmax(q_values[<span className="hljs-number">0</span>]){"\n"}{"\n"}{"        "}next_state, reward, done, _ = env.step(action){"\n"}{"        "}next_state = np.reshape(next_state, [<span className="hljs-number">1</span>, state_shape[<span className="hljs-number">0</span>]]){"\n"}{"        "}memory.append((state, action, reward, next_state, done)){"\n"}{"        "}<span className="hljs-keyword">if</span> <span className="hljs-built_in">len</span>(memory) &gt; max_memory:{"\n"}{"            "}memory.pop(<span className="hljs-number">0</span>){"\n"}{"\n"}{"        "}state = next_state{"\n"}{"        "}total_reward += reward{"\n"}{"\n"}{"        "}<span className="hljs-keyword">if</span> done:{"\n"}{"            "}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Episode: <span className="hljs-subst">{"{"}episode+<span className="hljs-number">1</span>{"}"}</span>, Reward: <span className="hljs-subst">{"{"}total_reward{"}"}</span>, Epsilon: <span className="hljs-subst">{"{"}epsilon:<span className="hljs-number">.2</span>f{"}"}</span>'</span>){"\n"}{"            "}<span className="hljs-keyword">break</span>{"\n"}{"\n"}{"        "}<span className="hljs-keyword">if</span> <span className="hljs-built_in">len</span>(memory) &gt; batch_size:{"\n"}{"            "}minibatch = random.sample(memory, batch_size){"\n"}{"            "}<span className="hljs-keyword">for</span> state_batch, action_batch, reward_batch, next_state_batch, done_batch <span className="hljs-keyword">in</span> minibatch:{"\n"}{"                "}target = reward_batch{"\n"}{"                "}<span className="hljs-keyword">if</span> <span className="hljs-keyword">not</span> done_batch:{"\n"}{"                    "}target += gamma * np.amax(model.predict(next_state_batch)[<span className="hljs-number">0</span>]){"\n"}{"                "}target_f = model.predict(state_batch){"\n"}{"                "}target_f[<span className="hljs-number">0</span>][action_batch] = target{"\n"}{"                "}model.fit(state_batch, target_f, epochs=<span className="hljs-number">1</span>, verbose=<span className="hljs-number">0</span>){"\n"}{"\n"}{"    "}<span className="hljs-keyword">if</span> epsilon &gt; epsilon_min:{"\n"}{"        "}epsilon *= epsilon_decay{"\n"}{"\n"}<span className="hljs-comment"># Evaluate the DQN</span>{"\n"}total_rewards = <span className="hljs-number">0</span>{"\n"}<span className="hljs-keyword">for</span> _ <span className="hljs-keyword">in</span> <span className="hljs-built_in">range</span>(<span className="hljs-number">100</span>):{"\n"}{"    "}state = env.reset(){"\n"}{"    "}state = np.reshape(state, [<span className="hljs-number">1</span>, state_shape[<span className="hljs-number">0</span>]]){"\n"}{"    "}done = <span className="hljs-literal">False</span>{"\n"}{"    "}<span className="hljs-keyword">while</span> <span className="hljs-keyword">not</span> done:{"\n"}{"        "}action = np.argmax(model.predict(state)[<span className="hljs-number">0</span>]){"\n"}{"        "}state, reward, done, _ = env.step(action){"\n"}{"        "}state = np.reshape(state, [<span className="hljs-number">1</span>, state_shape[<span className="hljs-number">0</span>]]){"\n"}{"        "}total_rewards += reward{"\n"}{"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Average Reward: <span className="hljs-subst">{"{"}total_rewards / <span className="hljs-number">100</span>{"}"}</span>'</span>){"\n"}</code></div></pre></div>
              <h4>Policy Gradient Methods</h4>
              <p>Policy gradient methods directly optimize the
                policy by adjusting the parameters of the
                policy network to maximize the expected
                cumulative reward. These methods are
                suitable for environments with large or
                continuous action spaces.</p>
              <p><strong>Example: Policy Gradient with
                  TensorFlow</strong></p>
              <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">import</span> tensorflow <span className="hljs-keyword">as</span> tf{"\n"}<span className="hljs-keyword">from</span> tensorflow.keras <span className="hljs-keyword">import</span> layers{"\n"}<span className="hljs-keyword">import</span> numpy <span className="hljs-keyword">as</span> np{"\n"}<span className="hljs-keyword">import</span> gym{"\n"}{"\n"}<span className="hljs-comment"># Create the environment</span>{"\n"}env = gym.make(<span className="hljs-string">'CartPole-v1'</span>){"\n"}{"\n"}<span className="hljs-comment"># Define the policy network</span>{"\n"}<span className="hljs-keyword">def</span> <span className="hljs-title function_">build_policy_network</span>(<span className="hljs-params">state_shape, action_shape</span>):{"\n"}{"    "}model = tf.keras.Sequential([{"\n"}{"        "}layers.Dense(<span className="hljs-number">24</span>, activation=<span className="hljs-string">'relu'</span>, input_shape=state_shape),{"\n"}{"        "}layers.Dense(<span className="hljs-number">24</span>, activation=<span className="hljs-string">'relu'</span>),{"\n"}{"        "}layers.Dense(action_shape, activation=<span className="hljs-string">'softmax'</span>){"\n"}{"    "}]){"\n"}{"    "}model.<span className="hljs-built_in">compile</span>(optimizer=tf.keras.optimizers.Adam(learning_rate=<span className="hljs-number">0.001</span>), loss=<span className="hljs-string">'categorical_crossentropy'</span>){"\n"}{"    "}<span className="hljs-keyword">return</span> model{"\n"}{"\n"}<span className="hljs-comment"># Hyperparameters</span>{"\n"}gamma = <span className="hljs-number">0.99</span>{"\n"}episodes = <span className="hljs-number">1000</span>{"\n"}{"\n"}<span className="hljs-comment"># Initialize model</span>{"\n"}state_shape = (env.observation_space.shape[<span className="hljs-number">0</span>],){"\n"}action_shape = env.action_space.n{"\n"}model = build_policy_network(state_shape, action_shape){"\n"}{"\n"}<span className="hljs-comment"># Helper function to choose an action based on the policy</span>{"\n"}<span className="hljs-keyword">def</span> <span className="hljs-title function_">choose_action</span>(<span className="hljs-params">model, state</span>):{"\n"}{"    "}state = np.reshape(state, [<span className="hljs-number">1</span>, state_shape[<span className="hljs-number">0</span>]]){"\n"}{"    "}probabilities = model.predict(state)[<span className="hljs-number">0</span>]{"\n"}{"    "}<span className="hljs-keyword">return</span> np.random.choice(action_shape, p=probabilities){"\n"}{"\n"}<span className="hljs-comment"># Helper function to discount rewards</span>{"\n"}<span className="hljs-keyword">def</span> <span className="hljs-title function_">discount_rewards</span>(<span className="hljs-params">rewards, gamma</span>):{"\n"}{"    "}discounted = np.zeros_like(rewards){"\n"}{"    "}cumulative = <span className="hljs-number">0</span>{"\n"}{"    "}<span className="hljs-keyword">for</span> t <span className="hljs-keyword">in</span> <span className="hljs-built_in">reversed</span>(<span className="hljs-built_in">range</span>(<span className="hljs-built_in">len</span>(rewards))):{"\n"}{"        "}cumulative = cumulative * gamma + rewards[t]{"\n"}{"        "}discounted[t] = cumulative{"\n"}{"    "}<span className="hljs-keyword">return</span> discounted{"\n"}{"\n"}<span className="hljs-comment"># Training the policy network</span>{"\n"}<span className="hljs-keyword">for</span> episode <span className="hljs-keyword">in</span> <span className="hljs-built_in">range</span>(episodes):{"\n"}{"    "}state = env.reset(){"\n"}{"    "}states, actions, rewards = [], [], []{"\n"}{"    "}total_reward = <span className="hljs-number">0</span>{"\n"}{"\n"}{"    "}<span className="hljs-keyword">while</span> <span className="hljs-literal">True</span>:{"\n"}{"        "}action = choose_action(model, state){"\n"}{"        "}next_state, reward, done, _ = env.step(action){"\n"}{"        "}states.append(state){"\n"}{"        "}actions.append(action){"\n"}{"        "}rewards.append(reward){"\n"}{"        "}state = next_state{"\n"}{"        "}total_reward += reward{"\n"}{"\n"}{"        "}<span className="hljs-keyword">if</span> done:{"\n"}{"            "}discounted_rewards = discount_rewards(rewards, gamma){"\n"}{"            "}discounted_rewards -= np.mean(discounted_rewards){"\n"}{"            "}discounted_rewards /= np.std(discounted_rewards){"\n"}{"            "}{"\n"}{"            "}<span className="hljs-keyword">for</span> i <span className="hljs-keyword">in</span> <span className="hljs-built_in">range</span>(<span className="hljs-built_in">len</span>(states)):{"\n"}{"                "}state_batch = np.reshape(states[i], [<span className="hljs-number">1</span>, state_shape[<span className="hljs-number">0</span>]]){"\n"}{"                "}action_batch = np.zeros([<span className="hljs-number">1</span>, action_shape]){"\n"}{"                "}action_batch[<span className="hljs-number">0</span>][actions[i]] = <span className="hljs-number">1</span>{"\n"}{"                "}reward_batch = discounted_rewards[i]{"\n"}{"                "}model.fit(state_batch, action_batch, sample_weight=[reward_batch], verbose=<span className="hljs-number">0</span>){"\n"}{"            "}{"\n"}{"            "}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Episode: <span className="hljs-subst">{"{"}episode+<span className="hljs-number">1</span>{"}"}</span>, Reward: <span className="hljs-subst">{"{"}total_reward{"}"}</span>'</span>){"\n"}{"            "}<span className="hljs-keyword">break</span>{"\n"}{"\n"}<span className="hljs-comment"># Evaluate the policy network</span>{"\n"}total_rewards = <span className="hljs-number">0</span>{"\n"}<span className="hljs-keyword">for</span> _ <span className="hljs-keyword">in</span> <span className="hljs-built_in">range</span>(<span className="hljs-number">100</span>):{"\n"}{"    "}state = env.reset(){"\n"}{"    "}done = <span className="hljs-literal">False</span>{"\n"}{"    "}<span className="hljs-keyword">while</span> <span className="hljs-keyword">not</span> done:{"\n"}{"        "}action = choose_action(model, state){"\n"}{"        "}state, reward, done, _ = env.step(action){"\n"}{"        "}total_rewards += reward{"\n"}{"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Average Reward: <span className="hljs-subst">{"{"}total_rewards / <span className="hljs-number">100</span>{"}"}</span>'</span>){"\n"}</code></div></pre></div>
            <h4>Advanced Reinforcement Learning Algorithms
            </h4>
            <p>Several advanced reinforcement learning
              algorithms have been developed to improve
              performance and stability. Some notable
              algorithms include:</p>
            <ol>
              <li><strong>Deep Deterministic Policy
                  Gradient (DDPG)</strong>: An
                algorithm for continuous action spaces
                that combines DQNs and policy gradient
                methods.</li>
              <li><strong>Proximal Policy Optimization
                  (PPO)</strong>: An on-policy
                algorithm that improves training
                stability by limiting policy updates.
              </li>
              <li><strong>A3C (Asynchronous Advantage
                  Actor-Critic)</strong>: An
                asynchronous version of the actor-critic
                method that uses multiple parallel
                workers to stabilize training.</li>
            </ol>
            <p><strong>Example: Using Stable Baselines for
                PPO</strong></p>
            <pre><div className="overflow-y-auto p-4 text-left undefined" dir="ltr"><code className="!whitespace-pre hljs language-python"><span className="hljs-keyword">from</span> stable_baselines3 <span className="hljs-keyword">import</span> PPO{"\n"}<span className="hljs-keyword">from</span> stable_baselines3.common.envs <span className="hljs-keyword">import</span> DummyVecEnv{"\n"}{"\n"}<span className="hljs-comment"># Create the environment</span>{"\n"}env = DummyVecEnv([<span className="hljs-keyword">lambda</span>: gym.make(<span className="hljs-string">'CartPole-v1'</span>)]){"\n"}{"\n"}<span className="hljs-comment"># Initialize the PPO agent</span>{"\n"}model = PPO(<span className="hljs-string">'MlpPolicy'</span>, env, verbose=<span className="hljs-number">1</span>){"\n"}{"\n"}<span className="hljs-comment"># Train the agent</span>{"\n"}model.learn(total_timesteps=<span className="hljs-number">100000</span>){"\n"}{"\n"}<span className="hljs-comment"># Evaluate the agent</span>{"\n"}total_rewards = <span className="hljs-number">0</span>{"\n"}episodes = <span className="hljs-number">100</span>{"\n"}<span className="hljs-keyword">for</span> _ <span className="hljs-keyword">in</span> <span className="hljs-built_in">range</span>(episodes):{"\n"}{"    "}state = env.reset(){"\n"}{"    "}done = <span className="hljs-literal">False</span>{"\n"}{"    "}<span className="hljs-keyword">while</span> <span className="hljs-keyword">not</span> done:{"\n"}{"        "}action, _ = model.predict(state){"\n"}{"        "}state, reward, done, _ = env.step(action){"\n"}{"        "}total_rewards += reward{"\n"}{"\n"}<span className="hljs-built_in">print</span>(<span className="hljs-string">f'Average Reward: <span className="hljs-subst">{"{"}total_rewards / episodes{"}"}</span>'</span>){"\n"}</code></div></pre></div>
          <h4>Practical Applications of Reinforcement
            Learning</h4>
          <p>Reinforcement learning has numerous practical
            applications across various domains. Here
            are some examples:</p>
          <ol>
            <li><strong>Robotics</strong>: Training
              robots to perform complex tasks, such as
              assembly, navigation, and manipulation.
            </li>
            <li><strong>Gaming</strong>: Developing AI
              agents that can play and master video
              games.</li>
            <li><strong>Finance</strong>: Optimizing
              trading strategies and portfolio
              management.</li>
            <li><strong>Healthcare</strong>:
              Personalizing treatment plans and
              optimizing drug dosing.</li>
            <li><strong>Autonomous Vehicles</strong>:
              Enabling self-driving cars to navigate
              and make decisions in real-time.</li>
          </ol>
          <h4>Practical Tips for Reinforcement Learning
          </h4>
          <p>Here are some practical tips to improve your
            reinforcement learning projects:</p>
          <ol>
            <li><strong>Start Simple</strong>: Begin
              with simpler environments and algorithms
              to understand the basics before moving
              on to more complex tasks.</li>
            <li><strong>Tune Hyperparameters</strong>:
              Experiment with different
              hyperparameters to find the best
              configuration for your task.</li>
            <li><strong>Use Visualization
                Tools</strong>: Visualize the
              training process and agent behavior to
              gain insights and diagnose issues.</li>
            <li><strong>Leverage Pre-trained
                Models</strong>: Use pre-trained
              models and transfer learning to
              accelerate training and improve
              performance.</li>
            <li><strong>Monitor and Evaluate</strong>:
              Regularly monitor and evaluate the
              agent's performance to ensure it is
              learning effectively.</li>
          </ol>
          <h4>Conclusion</h4>
          <p>Reinforcement learning (RL) is a powerful
            machine learning paradigm for training
            agents to make decisions and perform tasks
            through trial and error. This chapter
            covered the fundamentals of RL, including
            key concepts, Q-Learning, Deep Q-Networks
            (DQNs), Policy Gradient methods, and
            advanced algorithms like PPO. By
            understanding and applying these techniques,
            you can build robust RL agents that achieve
            high performance on a wide range of
            decision-making tasks.</p>
        </div>
      </div>
    </div>
  </div>
    )
    

}

export default ChapterFifteenth