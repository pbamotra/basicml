---
layout: post
title: KL Divergence and its links to entropy
categories: 
    - performance
tags: machine-learning statistics information-theory
cover_art: url(/assets/imgs/kldivergence.jpg) no-repeat right
cover_art_size: 100%
cover_attribution: flickr.com/photos/shonk
---

KL divergence is a [premetric](https://en.wikipedia.org/wiki/Metric_(mathematics)#Premetrics) that finds its root in information theory. It has a close relationship with [Shannon entropy](https://en.wiktionary.org/wiki/Shannon_entropy) and we'll walk through this relationship in the subsequent discussion. In its most basic sense, KL divergence measures the proximity between distributions. When we talk about KL divergence between two distribution say P and Q, it's denoted as

$$D_{KL} \left(P  \Vert  Q\right)$$

### Mathematical background

<p>
KL divergence belongs to a class of divergence measures known as <i>f-divergence</i>. For distributions \( P \) and \( Q \) and a convex function \( f(t) \) defined over \( t \gt 0 \) with \( f(1) = 0 \) is given by
</p>

$$D_{f} \left(P  \Vert Q\right) = Q(t) f\Big(\frac{P(t)}{Q(t)}\Big)$$

<p>
To derive KL divergence we set \( f(t) = t \ \log \left( t \right) \). For \( P(t) = Q(t) = 0 \), f-divergence is taken as zero. As per literature, KL divergence \( D_{KL} \left(P  \Vert  Q\right) \) requires P to be <i>absolute continuous</i>. Mathematically, this would mean KL divergence is undefined when for any t, P(t) \( \neq \) 0 but Q(t) = 0. An intuitive explanation for this will be presented later.
</p>

<p>
Three important properties of KL divergence are:-
<ul>
	<li> \( D_{KL} \left(P  \Vert  Q\right) \geq 0 \) . The equality happens when P = Q everywhere. This is known as Gibbs inequality. </li>
	<li> In general, \( D_{KL} \left(P  \Vert  Q\right) \neq D_{KL} \left(Q  \Vert  P\right) \). That means KL divergence is not symmetric and hence is not a metric/distance measure. </li>
	<li> KL divergence doesn't obey triangle inequality. </li>
</ul>
</p>

### Shannon entropy 

<p>
In computer science theory, entropy is one of the most studied topics. Thanks to Claude Shannon who gave us Shannon entropy. For a random variable \( X \) with PMF \( P(X) \), Shannon entropy is defined as
</p>

<p>
$$
H(X) = - \sum_{x}P(x)\log_{2}\left(P(x)\right)
$$
</p>

Intuitively, entropy gives us the lower bound on the number of bits required to optimally encode each observation of x [^1]. However, it must be kept in mind that we don't get to know what the optimal encoding is! The choice of use logarithm base 2 comes from information theory literature leading to entropy's unit as bits.

### KL divergence and its relationship with entropy

<p>
We saw that KL divergence is defined as \( D_{KL} \left(P  \Vert  Q\right) = \sum_{x} P(x) \log \Big( \frac{ P(x) }{ Q(x) } \Big) \). Let's rewrite this by expanding the log term. We get,
</p>

<p>
$$
\begin{aligned}
D_{KL} \left(P  \Vert Q\right) &= \sum_{x} P(x) \log\Big(\frac{P(x)}{Q(x)}\Big) \\
&= \sum_{x} P(x) \log(P(x)) - \sum_{x} P(x) \log(Q(x)) \\
&= -H(X) + H(P, Q)
\end{aligned}
$$
</p>

<p>
The two terms in the final step are well known. \( H(X) \) is the Shannon entropy which we described in the previous section. \( H(P, Q) \) is, yeah you probably guessed it, cross-entropy. Using Gibbs inequality, we can say that cross entropy is always greater than or equal to the corresponding Shannon entropy. 
</p>

<p>
Now, we describe KL divergence in terms of Shannon entropy and cross-entropy. Shannon entropy as we said above is the minimum number of bits required to optimally encode a distribution. Cross-entropy \( H(P, Q) \) on the other hand is the number of bits required to encode distribution P using an encoding that's optimal for distribution \( Q \) but not for \( P \). Consequently, KL divergence is the expected number of extra bits that are used under this sub-optimal encoding. 
</p>

<p>
Let's revisit the discussion on why we require P to be <i>absolute continuous</i>. Having \( Q(x) = 0 \) when \( P(x) \neq 0 \) would mean that we're trying to approximate a <i>probable</i> event with something that's definitely not going to happen. So, when such an event happens (in distribution P), KL divergence would essentially diverge logarithmically. In other words, the sub-optimal encoding has no way to encode such an event! So, KL divergence is undefined in such a case.
</p>

### Treading to machine learning domain

In most of the ML algorithms, we resort to optimising cross entropy and not KL divergence because the Shannon entropy term is independent of the model parameters and acts like a constant when taking derivative of log-likelihood. In fact, it can be shown that minimizing KL divergence is equivalent to minimizing negative log-likelihood.

<p>
Let \( p\left(x \vert \theta^{*}\right) \) be the true data distribution and model distribution be  \( p\left(x \vert \theta \right) \). Then by definition of KL divergence,
</p>

<p>
$$
\begin{aligned}
D_{KL}[p(x \vert \theta^*) \, \Vert \, p(x \vert \theta)] &= \mathbb{E}_{x \sim p(x \vert \theta^*)}\left[\log \frac{p(x \vert \theta^*)}{p(x \vert \theta)} \right] \\
&= \mathbb{E}_{x \sim p(x \vert \theta^*)}\left[\log \, p(x \vert \theta^*) - \log \, p(x \vert \theta) \right] \\
&= H(X) - \mathbb{E}_{x \sim p(x \vert \theta^*)}\left[\log \, p(x \vert \theta) \right]
\end{aligned}
$$
</p>

<p>
For a large number of samples drawn from the true distribution we have \( \frac{1}{N} \sum_x \log \, p(x \vert \theta) = \mathbb{E}_{x \sim p(x \vert \theta^*)}\left[\log \, p(x \vert \theta) \right] \) using the law of large numbers. Left-hand side in the equation represents log-likelihood of data samples. Comparing this result with the derivation above we can conclude that minimizing KL divergence is equivalent to minimizing negative log-likelihood.
</p>

These results have been used in variational inference theory and the most recent examples are Variational Autoencoders. The discussion about VAEs is reserved for another post. But you can read about them in this [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf) by [Carl Doersch](http://www.carldoersch.com/).

[^1]: [CMU 15-359: Elements of Information Theory](http://www.cs.cmu.edu/~venkatg/teaching/ITCS-spr2013/notes/15359-2009-lecture25.pdf)