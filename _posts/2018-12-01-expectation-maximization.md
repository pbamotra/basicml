---
layout: post
title: Expectation Maximization distilled to basics
categories: 
    - basics
tags: machine-learning expectation-maximization latent-models
cover_art: url(/assets/imgs/icons8/flamenco/flamenco-searching.png) no-repeat right
cover_art_size: 100%
cover_attribution: icons8.com/ouch
---
<p>
This article is about the Expectation Maximization algorithm and the guarantees it offers for certain kind of optimization problems. We'll walk through the gory mathematical details and work out some examples that involve EM. In this article, we assume that you're familier with maximum likelihood estimation.
</p>

> The expectation maximization algorithm enables parameter estimation in probabilistic models with incomplete data.

Don't worry even if you didn't understand the previous statement. But, keep in mind the three terms - <i>parameter estimation</i><i>, probabilistic models</i>, and <i>incomplete data</i> because this is what the EM is all about. So, hold on tight.

### How EM differs from MLE?
<p>
Given the observed data \( \mathcal{y} \) with density \( P ( y \vert \theta ) \) for \( \theta \in \Omega \). For MLE, using only \( \mathcal{y} \) we solve parameter estimation as,
</p>

<p>
\[
\hat{\theta}_{\text{mle}} = \text{arg max}_{\theta \in \Omega} \log \mathcal{p}(\mathcal{y}, \theta)
\]
</p>

<p>
However, in the case of EM, we have a concept of complete data, \( \mathrm{X} \). This complete data is not observed directly. The only observed data is \( \mathrm{y} \) represented by random variable \( \mathrm{Y} \) which depends on \( \mathrm{X} \). E.g. Y can represent the mean of X or Y could be the first component of vector \( \mathrm{X} \). In EM, we try to find the <i>expected log-likelihood</i> of this complete data \( \mathrm{X} \). 
</p> 

<br/>
EM algorithm can be summarized in following five steps[^1] :-
<br/>

<ol>
    <li>Let m = 0 and make an initial estimate \( \theta^{( m )} \) for \( \theta \).</li>
    <li>Given the observed data y and pretending for the moment that your current guess \( \theta^{( m )} \) is correct, formulate the conditional probability distribution \( \mathcal{p(} \mathcal{x} \vert \mathcal{y}, \theta (m) ) \) for the complete data \( \mathcal{x} \).</li>
    <li>Using the conditional probability distribution \( \mathcal{p(} \mathcal{x} \vert \mathcal{y}, \theta (m) ) \) calculated in the previous step, form the conditional expected log-likelihood, called the Q-function.
        <p>
            \[
            \begin{aligned}
            Q ( \theta | \theta ^ { ( m ) } ) &= \int _ { \mathcal{X} (y) } \operatorname{\log} p ( x | \theta ) p ( x | y ,\theta ^ { ( m ) } ) d x \\\
            &= E _ { X | y ,\theta ( m ) } [ \operatorname{\log} p ( X | \theta ) ]
            \end{aligned}
            \]
        </p>
        
    \( \mathcal{X} (y) \) denotes support of \( \mathrm{X} \), which is the set \( \{ {\mathcal{x} \vert \mathcal{p}(\mathcal{x} \vert \theta) > 0} \} \). Also, \( \mathcal{X} (y) \) does not depend on \( \theta \). If the support depends on \( \theta \) (e.g. unif(0, \( \theta) \)) then the monotonicity of the EM algorithm might not hold. We'll talk about monotonicity of EM soon.
    </li>
    <li>Find the \( \theta \) that maximizes the Q-function; the result is your new estimate \( \theta^{(m+1)} \).</li>
    <li>Stop when the \( \vert \mathcal{l}( \theta^{(m+1)} ) - \mathcal{l} ( \theta^{(m)}) \vert \lt \epsilon \) where \( \mathcal{l} \) represents the log-likelihood and \( \epsilon \gt 0 \).</li>
</ol>

### EM coin toss example
#### Problem statement
Source: [Do et. al., What is the expectation maximization
algorithm?](https://pdfs.semanticscholar.org/a5c4/35690c717d04801a68950f14036c38f2a9ab.pdf)
<br/>
<br/>

<p>
Consider a simple coin-flipping experiment in which we are given a pair of coins A and B of unknown biases, \( \theta_{A} \) and
\( \theta_{B} \), respectively (that is, on any given flip, coin A will land on heads with probability \( \theta_{A} \) and tails with probability 1 – \( \theta_{A} \) and similarly for coin B). Our goal is to estimate \( \theta \) = (\( \theta_{A} \), \( \theta_{B} \)) by repeating the following procedure five times: randomly choose one of the two coins (with equal probability), and perform ten independent coin tosses with the selected coin. Thus, the entire procedure involves a total of 50 coin tosses. 
</p>

<div class="svg-container">
    <figure class="caption">
        <img src="/assets/imgs/nature-em.svg" alt="EM coin toss example"/>
        <figcaption>Source: Nature Biotechnology, Vol. 26 Page: 898</figcaption>
    </figure> 
</div>


#### EM Solution
<p>
If in this problem, we knew which coin was used in each of the five experiments then we can simply resort to using MLE as shown in Figure 1(a). However, what do we do in the case we don't know which coin was used for each of the experiment as shown by <b>?</b> coin symbol (in gray) in Figure 1(b). This is where use EM to model the hidden data and come up with good estimates of our parameters \( \theta \). 
</p>

<br/>
Now let's define our EM steps for this problem: -
<br/>
<br/>

<ol>
    <li>Let \( \theta^{(0)} \) = (0.60, 0.50). Let X = (Y, Z), where Y is the observed coin tosses and Z represent the coin that was chosen for the experiment. This will be our definition of complete data in this case. In the subsequent discussion, we'll use \( \mathcal{y}_{j} \) to denote the \( \text{j}^{\text{th}}\) experiment. Also, Z will treated as an indicator variable \( \mathbb{I}_{Z} \) such that \( \mathbb{I}_{Z} \) = 1 if coin A was chosen for an experiment and \( \mathbb{I}_{Z} \) = 0 if coin B was chosen.</li>
    
    <li>
    Next, we define conditional probability distribution \( \mathcal{p(} \mathcal{x} \vert \mathcal{y}, \theta (m) ) \). Since, \( \mathcal{y} \) is subsumed into \( \mathcal{x} \) we can conveniently re-write this conditional probability as \( \mathcal{p(} \mathcal{y}, \mathcal{z} \vert \theta (m) ) \). Let's expand this term and break it into different components. It going to be math-heavy so pay attention to each step.
    
    $$
    \begin{aligned}
        \mathcal{p(} \mathcal{y}, \mathcal{z} \vert \theta (m) ) 
        &= \prod_{j} \mathcal{p(} \mathcal{y_{j}}, \mathcal{z_{j}} \vert \theta (m) && \text{[independence]} \\\
        \log \mathcal{p(} \mathcal{y}, \mathcal{z} \vert \theta^{(m)} )) 
        
        &= \log \prod_{j} \mathcal{p(} \mathcal{y_{j}}, \mathcal{z_{j}} \vert \theta^{(m)}) && \text{[simplicity]} \\\
        
        &= \sum_{j} \log \mathcal{p(} \mathcal{y_{j}}, \mathcal{z_{j}} \vert \theta^{(m)}) \\\
        
        &= \sum_{j} \log [ \mathcal{p(} \mathcal{y_{j}}, \mathcal{z_{j}}=1 \vert \theta^{(m)})^{z_j} \\\
        & \hspace{1.5cm} \times \mathcal{p(} \mathcal{y_{j}}, \mathcal{z_{j}}=0 \vert \theta^{(m)})^{1 - z_{j}} ] \\\
        
        &= \sum_{j} \log \Bigg[ \Big(\frac{1}{2} \text{ } \binom{10}{h_{j}} \text{ } \theta_{A}^{h_{j}} \text{ } (1 - \theta_{A})^{10 - h_{j}} \Big)^{z_{j}} \\\
        &  \hspace{1.5cm} \times \Big(\frac{1}{2} \text{ } \binom{10}{h_{j}} \text{ } \theta_{B}^{h_{j}} \text{ } (1 - \theta_{B})^{10 -h_{j}} \Big)^{1 - z_{j}} \Bigg] && \text{[$ h_{j} $ = #heads]} \\\
        
        &= \sum_{j} \log \Bigg[ \Big( \mathcal{C} \text{ } \theta_{A}^{h_{j}} \text{ } (1 - \theta_{A})^{10-h_{j}} \Big)^{z_{j}} \\\
        &  \hspace{1.5cm} \times \Big( \mathcal{C} \text{ } \theta_{B}^{h_{j}} \text{ } (1 - \theta_{B})^{10-h_{j}} \Big)^{1 - z_{j}} \Bigg] \\\
        
        &= \sum_{j} \Bigg[ z_{j} \log \mathcal{C} \\\
        &  \hspace{1cm} + z_{j} \text{ } h_{j} \log \theta_{A} \\\
        &  \hspace{1cm} + z_{j} \text{ } (10 - h_{j}) \log (1 - \theta_{A})  \\\
        &  \hspace{1cm} + (1 - z_{j}) \log \mathcal{C} \\\
        &  \hspace{1cm} + (1 - z_{j}) \text{ } h_{j} \log \theta_{B} \\\
        &  \hspace{1cm} + (1 - z_{j}) \text{ } (10 - h_{j}) \log (1 - \theta_{B}) \Bigg] \\\
    \end{aligned}
    $$
    </li>
    <li><b>E-step:</b> \( Q = E _ { X | y ,\theta ( m ) } [ \operatorname{\log} p ( X | \theta ) ] \). In our case X = (Y, Z). Since Y is known the expectation that we want is w.r.t. to z. So, 
    
    $$
    \begin{aligned}
        Q 
        &= E _ { X | y ,\theta ( m ) } [ \operatorname{\log} p ( X | \theta ) ] \\\
        &= E _ { Z | \theta ( m ) } [ \operatorname{\log} p ( X | \theta ) ] \\\
        &= E _ { Z | \theta ( m ) } [ \operatorname{\log} p ( \mathcal{y}, \mathcal{z} | \theta ) ] \\\
        &= E _ { Z | \theta ( m ) } \Bigg[ \sum_{j} [ z_{j} \log \mathcal{C} \\\
        &  \hspace{1cm} + z_{j} \text{ } h_{j} \log \theta_{A} \\\
        &  \hspace{1cm} + z_{j} \text{ } (10 - h_{j}) \log (1 - \theta_{A})  \\\
        &  \hspace{1cm} + (1 - z_{j}) \log \mathcal{C} \\\
        &  \hspace{1cm} + (1 - z_{j}) \text{ } h_{j} \log \theta_{B} \\\
        &  \hspace{1cm} + (1 - z_{j}) \text{ } (10 - h_{j}) \log (1 - \theta_{B})] \Bigg] \\\
    \end{aligned}
    $$
    
    In all the terms involved in our Q-function, the only expectation we essentially need is \( E _ { Z | \theta ( m ) } [z_j]\).
    \( Z \) is an indicator variable and \( \mathcal{p} (\mathbb{I}_{Z}) = p(z=1) \), we now find \( p(z=1) \).
    
    $$
    \begin{aligned}
        P(z_{j}=1 \vert y_{j}, \theta^{(m)})
        &= P(\text{using coin A} \vert y_{j}, \theta^{(m)}) \\\
        &= \frac{P(y_{j} \vert \text{using coin A}) \text{ } P(\text{using coin A})}{P(y_j)} \\\
        &= \frac{P(y_{j} \vert \text{A}) \text{ } P(\text{A})}{P(y_{j} \vert \text{A}) \text{ } P(\text{A}) + P(y_{j} \vert \text{B}) \text{ } P(\text{B})}   \\\
        &= \frac{(\binom{10}{h_j} \text{ } \theta_{A}^{h_j} \text{ } (1 - \theta_{A})^{10-h_j}) \text{ } (\frac{1}{2})}
                {(\binom{10}{h_j} \text{ } \theta_{A}^{h_j} \text{ } (1 - \theta_{A})^{10-h_j}) \text{ } (\frac{1}{2}) + 
                 (\binom{10}{h_j} \text{ } \theta_{B}^{h_j} \text{ } (1 - \theta_{B})^{10-h_j}) \text{ } (\frac{1}{2})} \\\
        E _ { Z | \theta ( m ) } [z_j]
        &= \frac{(\theta_{A}^{h_j} \text{ } (1 - \theta_{A})^{10-h_j})}
                {(\theta_{A}^{h_j} \text{ } (1 - \theta_{A})^{10-h_j}) + 
                 (\theta_{B}^{h_j} \text{ } (1 - \theta_{B})^{10-h_j})} \\\
        &= P_{j}
    \end{aligned}
    $$
    Let's subsitute this finding into our Q-function.
    $$
    \begin{aligned}
        Q
        &= E _ { Z | \theta ( m ) } \Bigg[ \sum_{j} [ z_{j} \log \mathcal{C} \\\
        &  \hspace{1cm} + z_{j} \text{ } h_{j} \log \theta_{A} \\\
        &  \hspace{1cm} + z_{j} \text{ } (10 - h_{j}) \log (1 - \theta_{A})  \\\
        &  \hspace{1cm} + (1 - z_{j}) \log \mathcal{C} \\\
        &  \hspace{1cm} + (1 - z_{j}) \text{ } h_{j} \log \theta_{B} \\\
        &  \hspace{1cm} + (1 - z_{j}) \text{ } (10 - h_{j}) \log (1 - \theta_{B})] \Bigg] \\\
        
        &= \sum_{j} \Big[ P_{j} \log \mathcal{C} \\\
        &  \hspace{1cm} + P_{j} \text{ } h_{j} \log \theta_{A} \\\
        &  \hspace{1cm} + P_{j} \text{ } (10 - h_{j}) \log (1 - \theta_{A})  \\\
        &  \hspace{1cm} + (1 - P_{j}) \log \mathcal{C} \\\
        &  \hspace{1cm} + (1 - P_{j}) \text{ } h_{j} \log \theta_{B} \\\
        &  \hspace{1cm} + (1 - P_{j}) \text{ } (10 - h_{j}) \log (1 - \theta_{B})] \Big] \\\
    \end{aligned}
    $$
    Phew, that finishes defining our Q-function and the E-step. Here, E stands for finding conditional expected log-likelihood or simply our Q-function. Also, I know I haven't the numerical calculation in our problem statement. But, hang on, everything will fall into places soon.
    </li>
    <li><b>M-step</b>: Let's maximize our Q-function w.r.t \( \theta_A \text{ and } \theta_B \).
    $$
    \begin{aligned}
        \frac{\partial Q}{\partial \theta_A} 
        &= \sum_{j} \Big( \frac{P_{j} \text{ } h_j}{\theta_A} - \frac{P_{j} (10-h_j)}{(1 - \theta_A)} \Big)
        = 0
        \\\
        \frac{\partial Q}{\partial \theta_B} 
        &= \sum_{j} \Big( \frac{(1-P_{j}) \text{ } h_j}{\theta_B} - \frac{(1-P_{j}) (10-h_j)}{(1 - \theta_B)} \Big)
        = 0
    \end{aligned}
    $$
    
    On solving these two equations, we get
    
    $$
    \begin{aligned}
        \theta^{(m+1)}_{A} &= \frac{1}{10}\frac{\sum_{j} P_{j} h_j}{\sum_{j} P_{j}} && \text{[eq. 1]}\\\
        \theta^{(m+1)}_{B} &= \frac{1}{10}\frac{\sum_{j} (1 - P_{j}) h_j}{\sum_{j} (1 - P_{j})} && \text{[eq. 2]}
    \end{aligned}
    $$
    </li>
    <li>Let's solve our original problem now that we have all the pieces. <br/>
    $$
    \begin{aligned}
        & h_j = [5, 9, 8, 4, 7]\\\
        & \theta^{(0)} = (0.6, 0.5)\\\
        & \\\
        & P_{1} = \frac{(0.6)^{5} (0.4)^{5}}{(0.6)^{5} (0.4)^{5} + (0.5)^{5} (0.5)^{5}} \approx 0.45 \\\
        & P_{j} = [0.45, 0.80, 0.73, 0.35, 0.65] \\\
        & \sum_{j} P_{j} = 2.98 \\\
        & \\\
        & 1 - P_{j} = [1-0.45, 1-0.80, 1-0.73, 1-0.35, 1-0.65] \\\
        & \sum_{j} (1 - P_{j}) = 5 - 2.98 = 2.02 \\\
        & \\\
        & h_j P_{j} = [5*0.45, 9*0.80, 8*0.73, 4*0.35, 7*0.65] \\\
        & h_j (1 - P_{j}) = [5*0.55, 9*0.20, 8*0.27, 4*0.65, 7*0.35] \\\
        & \\\
        & \sum_{j} h_j P_{j} = 21.24 \\\
        & \sum_{j} h_j (1 - P_{j}) = 11.76
    \end{aligned}
    $$
    Substituting into eq. 1 and eq. 2, we get <br/>
    \( \theta^{(1)}_{A} = 0.71 \text{ and } \theta^{(1)}_{B} = 0.58 \).<br/>
    After 10 iterations, the algorithm will converge to \( \theta^{(1)}_{A} = 0.80 \text{ and } \theta^{(1)}_{B} = 0.52 \).
    <br/>
    Note: Some values are a little off from the ones in the figure 1 due of rounding off.
    </li>
</ol>

### What's happening under the hood?

#### Monotonicity
<p>
<b>Theorem: </b>Let random variables X and Y have parametric densities with parameter \( \theta \in \Omega \). Suppose the support of X does not depend on \( \theta \), and the Markov relationship \( \theta \rightarrow X \rightarrow Y \), that is, \( p(y |x, \theta) = p(y \vert x) \) holds for all \( \theta \in \Omega, x \in X \text{and} y \in Y \). Then for \( \theta \in \Omega \text{ and any } y \in Y \)
with \( \mathcal{X} (y) \neq \emptyset, \theta \geq (\theta^{(m)}) \) if \( Q(\theta \vert \theta^{(m)}) \geq Q(\theta^{(m)} \vert \theta^{(m)}) \).

</p>
<b>Proof:</b> see <a href="#fn1">[1]</a>. <br/>
<b>Geometric meaning: </b> The theorem states that improving Q-function at each step will not make the log-likelihood worse.

#### Geometric intuition behing E-step and M-step
<br/>
<div class="svg-container">
    <figure class="caption">
        <img src="/assets/imgs/em.png" alt="EM monotonicity"/>
        <figcaption>Inspired by: Sean Borman, EM Tutorial<sup><a href="#fn2">2</a></sup></figcaption>
    </figure> 
</div>
<br/>

<p>
The expectation maximization algorithm during the E-step, choose the Q-function such that it lower bounds \( \log P(x; \theta) \) everywhere, and for which \( Q(\theta^{(m)} \vert \theta^{(m)}) \) = \(\log P(x; \theta^{(m)}) \). During the M-step, the algorithm moves to a new parameter set \( \theta^{(m+1)} \) that maximizes \( Q \). The proof for this can be found in <a href="#fn2">[2]</a>. In other words, EM tries to maximize the lower bound given by Q-function using co-ordinate ascent<sup><a href="#fn3">3</a></sup>.
</p>

<span style="display:none">[^2][^3][^4]</span>
<br/>

<p>
Theoretically, EM provides linear convergence but it is not guaranteed to find global optima. Thus, it is common to run EM with different initialisations of the parameters \( \theta \) and choose the results that correspond to largest likelihood value. Other methods like Newton-Raphson can be used to find optimal parameters with sublinear convergence but Hessian computation for such methods would be another challenge. EM equips us with a simple and robust tool that is effective in case of poor initial guesses. Such poor guesses may lead to unstable inversions in Hessian based methods.
</p>

[^1]: [Gupta et. al., Theory and Use of the EM Algorithm](http://mayagupta.org/publications/EMbookGuptaChen2010.pdf)
[^2]: [Borman S., The Expectation Maximization Algorithm](http://www.seanborman.com/publications/EM_algorithm.pdf)
[^3]: [Singh A., The EM Algorithm](https://www.cs.cmu.edu/~awm/15781/assignments/EM.pdf)
[^4]: [greeness, Expectation Maximization](https://github.com/greeness/EM-Tutorial/blob/master/document.pdf)