#!/usr/bin/env python
# -*-coding:utf-8-*-

import types
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from itertools import chain


def str2tuple(s, sep="/"):
    """
    Given the string representation of a tagged token, return the
    corresponding tuple representation.  The rightmost occurrence of
    *sep* in *s* will be used to divide *s* into a word string and
    a tag string.  If *sep* does not occur in *s*, return (s, None).

        >>> str2tuple('fly/NN')
        ('fly', 'NN')

    :type s: str
    :param s: The string representation of a tagged token.
    :type sep: str
    :param sep: The separator string used to separate word strings
        from tags.
    """
    loc = s.rfind(sep)
    if loc >= 0:
        return (s[:loc], s[loc + len(sep) :].upper())
    else:
        return (s, None)

def pad_sequence(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """
    Returns a padded sequence of items before ngram extraction.

        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']

    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence

def ngrams(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:

        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:

        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]


    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequence(
        sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol
    )

    history = []
    while n > 1:
        # PEP 479, prevent RuntimeError from being raised when StopIteration bubbles out of generator
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def bigrams(sequence, **kwargs):
    """
    Return the bigrams generated from a sequence of items, as an iterator.
    For example:

        >>> list(bigrams([1,2,3,4,5]))
        [(1, 2), (2, 3), (3, 4), (4, 5)]

    Use bigrams for a list version of this function.

    :param sequence: the source data to be converted into bigrams
    :type sequence: sequence or iter
    :rtype: iter(tuple)
    """

    for item in ngrams(sequence, 2, **kwargs):
        yield item


def _get_kwarg(kwargs, key, default):
    if key in kwargs:
        arg = kwargs[key]
        del kwargs[key]
    else:
        arg = default
    return arg

def raise_unorderable_types(ordering, a, b):
    raise TypeError(
        "unorderable types: %s() %s %s()"
        % (type(a).__name__, ordering, type(b).__name__)
    )

class FreqDist(Counter):
    def __init__(self, samples=None):
        """
        Construct a new frequency distribution.  If ``samples`` is
        given, then the frequency distribution will be initialized
        with the count of each object in ``samples``; otherwise, it
        will be initialized to be empty.

        In particular, ``FreqDist()`` returns an empty frequency
        distribution; and ``FreqDist(samples)`` first creates an empty
        frequency distribution, and then calls ``update`` with the
        list ``samples``.

        :param samples: The samples to initialize the frequency
            distribution with.
        :type samples: Sequence
        """
        Counter.__init__(self, samples)

        # Cached number of samples in this FreqDist
        self._N = None

    def N(self):
        """
        Return the total number of sample outcomes that have been
        recorded by this FreqDist.  For the number of unique
        sample values (or bins) with counts greater than zero, use
        ``FreqDist.B()``.

        :rtype: int
        """
        if self._N is None:
            # Not already cached, or cache has been invalidated
            self._N = sum(self.values())
        return self._N

    def __setitem__(self, key, val):
        """
        Override ``Counter.__setitem__()`` to invalidate the cached N
        """
        self._N = None
        super(FreqDist, self).__setitem__(key, val)

    def __delitem__(self, key):
        """
        Override ``Counter.__delitem__()`` to invalidate the cached N
        """
        self._N = None
        super(FreqDist, self).__delitem__(key)

    def update(self, *args, **kwargs):
        """
        Override ``Counter.update()`` to invalidate the cached N
        """
        self._N = None
        super(FreqDist, self).update(*args, **kwargs)

    def setdefault(self, key, val):
        """
        Override ``Counter.setdefault()`` to invalidate the cached N
        """
        self._N = None
        super(FreqDist, self).setdefault(key, val)

    def B(self):
        """
        Return the total number of sample values (or "bins") that
        have counts greater than zero.  For the total
        number of sample outcomes recorded, use ``FreqDist.N()``.
        (FreqDist.B() is the same as len(FreqDist).)

        :rtype: int
        """
        return len(self)

    def hapaxes(self):
        """
        Return a list of all samples that occur once (hapax legomena)

        :rtype: list
        """
        return [item for item in self if self[item] == 1]

    def Nr(self, r, bins=None):
        return self.r_Nr(bins)[r]

    def r_Nr(self, bins=None):
        """
        Return the dictionary mapping r to Nr, the number of samples with frequency r, where Nr > 0.

        :type bins: int
        :param bins: The number of possible sample outcomes.  ``bins``
            is used to calculate Nr(0).  In particular, Nr(0) is
            ``bins-self.B()``.  If ``bins`` is not specified, it
            defaults to ``self.B()`` (so Nr(0) will be 0).
        :rtype: int
        """

        _r_Nr = defaultdict(int)
        for count in self.values():
            _r_Nr[count] += 1

        # Special case for Nr[0]:
        _r_Nr[0] = bins - self.B() if bins is not None else 0

        return _r_Nr

    def _cumulative_frequencies(self, samples):
        """
        Return the cumulative frequencies of the specified samples.
        If no samples are specified, all counts are returned, starting
        with the largest.

        :param samples: the samples whose frequencies should be returned.
        :type samples: any
        :rtype: list(float)
        """
        cf = 0.0
        for sample in samples:
            cf += self[sample]
            yield cf

    # slightly odd nomenclature freq() if FreqDist does counts and ProbDist does probs,
    # here, freq() does probs
    def freq(self, sample):
        """
        Return the frequency of a given sample.  The frequency of a
        sample is defined as the count of that sample divided by the
        total number of sample outcomes that have been recorded by
        this FreqDist.  The count of a sample is defined as the
        number of times that sample outcome was recorded by this
        FreqDist.  Frequencies are always real numbers in the range
        [0, 1].

        :param sample: the sample whose frequency
               should be returned.
        :type sample: any
        :rtype: float
        """
        n = self.N()
        if n == 0:
            return 0
        return self[sample] / n

    def max(self):
        """
        Return the sample with the greatest number of outcomes in this
        frequency distribution.  If two or more samples have the same
        number of outcomes, return one of them; which sample is
        returned is undefined.  If no outcomes have occurred in this
        frequency distribution, return None.

        :return: The sample with the maximum number of outcomes in this
                frequency distribution.
        :rtype: any or None
        """
        if len(self) == 0:
            raise ValueError(
                "A FreqDist must have at least one sample before max is defined."
            )
        return self.most_common(1)[0][0]

    def plot(self, *args, **kwargs):
        """
        Plot samples from the frequency distribution
        displaying the most frequent sample first.  If an integer
        parameter is supplied, stop after this many samples have been
        plotted.  For a cumulative plot, specify cumulative=True.
        (Requires Matplotlib to be installed.)

        :param title: The title for the graph
        :type title: str
        :param cumulative: A flag to specify whether the plot is cumulative (default = False)
        :type title: bool
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ValueError(
                "The plot function requires matplotlib to be installed."
                "See http://matplotlib.org/"
            )

        if len(args) == 0:
            args = [len(self)]
        samples = [item for item, _ in self.most_common(*args)]

        cumulative = _get_kwarg(kwargs, "cumulative", False)
        percents = _get_kwarg(kwargs, "percents", False)
        if cumulative:
            freqs = list(self._cumulative_frequencies(samples))
            ylabel = "Cumulative Counts"
            if percents:
                freqs = [f / freqs[len(freqs) - 1] * 100 for f in freqs]
                ylabel = "Cumulative Percents"
        else:
            freqs = [self[sample] for sample in samples]
            ylabel = "Counts"
        # percents = [f * 100 for f in freqs]  only in ProbDist?

        ax = plt.gca()
        ax.grid(True, color="silver")

        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 2
        if "title" in kwargs:
            ax.set_title(kwargs["title"])
            del kwargs["title"]

        ax.plot(freqs, **kwargs)
        ax.set_xticks(range(len(samples)))
        ax.set_xticklabels([str(s) for s in samples], rotation=90)
        ax.set_xlabel("Samples")
        ax.set_ylabel(ylabel)

        plt.show()

        return ax

    def tabulate(self, *args, **kwargs):
        """
        Tabulate the given samples from the frequency distribution (cumulative),
        displaying the most frequent sample first.  If an integer
        parameter is supplied, stop after this many samples have been
        plotted.

        :param samples: The samples to plot (default is all samples)
        :type samples: list
        :param cumulative: A flag to specify whether the freqs are cumulative (default = False)
        :type title: bool
        """
        if len(args) == 0:
            args = [len(self)]
        samples = [item for item, _ in self.most_common(*args)]

        cumulative = _get_kwarg(kwargs, "cumulative", False)
        if cumulative:
            freqs = list(self._cumulative_frequencies(samples))
        else:
            freqs = [self[sample] for sample in samples]
        # percents = [f * 100 for f in freqs]  only in ProbDist?

        width = max(len("{}".format(s)) for s in samples)
        width = max(width, max(len("%d" % f) for f in freqs))

        for i in range(len(samples)):
            print("%*s" % (width, samples[i]), end=" ")
        print()
        for i in range(len(samples)):
            print("%*d" % (width, freqs[i]), end=" ")
        print()

    def copy(self):
        """
        Create a copy of this frequency distribution.

        :rtype: FreqDist
        """
        return self.__class__(self)

    # Mathematical operatiors

    def __add__(self, other):
        """
        Add counts from two counters.

        >>> FreqDist('abbb') + FreqDist('bcc')
        FreqDist({'b': 4, 'c': 2, 'a': 1})

        """
        return self.__class__(super(FreqDist, self).__add__(other))

    def __sub__(self, other):
        """
        Subtract count, but keep only results with positive counts.

        >>> FreqDist('abbbc') - FreqDist('bccd')
        FreqDist({'b': 2, 'a': 1})

        """
        return self.__class__(super(FreqDist, self).__sub__(other))

    def __or__(self, other):
        """
        Union is the maximum of value in either of the input counters.

        >>> FreqDist('abbb') | FreqDist('bcc')
        FreqDist({'b': 3, 'c': 2, 'a': 1})

        """
        return self.__class__(super(FreqDist, self).__or__(other))

    def __and__(self, other):
        """
        Intersection is the minimum of corresponding counts.

        >>> FreqDist('abbb') & FreqDist('bcc')
        FreqDist({'b': 1})

        """
        return self.__class__(super(FreqDist, self).__and__(other))

    def __le__(self, other):
        """
        Returns True if this frequency distribution is a subset of the other
        and for no key the value exceeds the value of the same key from
        the other frequency distribution.

        The <= operator forms partial order and satisfying the axioms
        reflexivity, antisymmetry and transitivity.

        >>> FreqDist('a') <= FreqDist('a')
        True
        >>> a = FreqDist('abc')
        >>> b = FreqDist('aabc')
        >>> (a <= b, b <= a)
        (True, False)
        >>> FreqDist('a') <= FreqDist('abcd')
        True
        >>> FreqDist('abc') <= FreqDist('xyz')
        False
        >>> FreqDist('xyz') <= FreqDist('abc')
        False
        >>> c = FreqDist('a')
        >>> d = FreqDist('aa')
        >>> e = FreqDist('aaa')
        >>> c <= d and d <= e and c <= e
        True
        """
        if not isinstance(other, FreqDist):
            raise_unorderable_types("<=", self, other)
        return set(self).issubset(other) and all(
            self[key] <= other[key] for key in self
        )

    def __ge__(self, other):
        if not isinstance(other, FreqDist):
            raise_unorderable_types(">=", self, other)
        return set(self).issuperset(other) and all(
            self[key] >= other[key] for key in other
        )

    __lt__ = lambda self, other: self <= other and not self == other
    __gt__ = lambda self, other: self >= other and not self == other

    def __repr__(self):
        """
        Return a string representation of this FreqDist.

        :rtype: string
        """
        return self.pformat()

    def pprint(self, maxlen=10, stream=None):
        """
        Print a string representation of this FreqDist to 'stream'

        :param maxlen: The maximum number of items to print
        :type maxlen: int
        :param stream: The stream to print to. stdout by default
        """
        print(self.pformat(maxlen=maxlen), file=stream)

    def pformat(self, maxlen=10):
        """
        Return a string representation of this FreqDist.

        :param maxlen: The maximum number of items to display
        :type maxlen: int
        :rtype: string
        """
        items = ["{0!r}: {1!r}".format(*item) for item in
                 self.most_common(maxlen)]
        if len(self) > maxlen:
            items.append("...")
        return "FreqDist({{{0}}})".format(", ".join(items))

    def __str__(self):
        """
        Return a string representation of this FreqDist.

        :rtype: string
        """
        return "<FreqDist with %d samples and %d outcomes>" % (
            len(self), self.N())

    def __iter__(self):
        """
        Return an iterator which yields tokens ordered by frequency.

        :rtype: iterator
        """
        for token, _ in self.most_common(self.B()):
            yield token


def _mro(cls):
    """
    Return the method resolution order for ``cls`` -- i.e., a list
    containing ``cls`` and all its base classes, in the order in which
    they would be checked by ``getattr``.  For new-style classes, this
    is just cls.__mro__.  For classic classes, this can be obtained by
    a depth-first left-to-right traversal of ``__bases__``.
    """
    if isinstance(cls, type):
        return cls.__mro__
    else:
        mro = [cls]
        for base in cls.__bases__:
            mro.extend(_mro(base))
        return mro


def overridden(method):
    """
    :return: True if ``method`` overrides some method with the same
    name in a base class.  This is typically used when defining
    abstract base classes or interfaces, to allow subclasses to define
    either of two related methods:

        >>> class EaterI:
        ...     '''Subclass must define eat() or batch_eat().'''
        ...     def eat(self, food):
        ...         if overridden(self.batch_eat):
        ...             return self.batch_eat([food])[0]
        ...         else:
        ...             raise NotImplementedError()
        ...     def batch_eat(self, foods):
        ...         return [self.eat(food) for food in foods]

    :type method: instance method
    """
    if isinstance(method,
                  types.MethodType) and method.__self__.__class__ is not None:
        name = method.__name__
        funcs = [
            cls.__dict__[name]
            for cls in _mro(method.__self__.__class__)
            if name in cls.__dict__
        ]
        return len(funcs) > 1
    else:
        raise TypeError("Expected an instance method.")


class TokenizerI(ABC):
    """
    A processing interface for tokenizing a string.
    Subclasses must define ``tokenize()`` or ``tokenize_sents()`` (or both).
    """

    @abstractmethod
    def tokenize(self, s):
        """
        Return a tokenized copy of *s*.

        :rtype: list of str
        """
        if overridden(self.tokenize_sents):
            return self.tokenize_sents([s])[0]

    def span_tokenize(self, s):
        """
        Identify the tokens using integer offsets ``(start_i, end_i)``,
        where ``s[start_i:end_i]`` is the corresponding token.

        :rtype: iter(tuple(int, int))
        """
        raise NotImplementedError()

    def tokenize_sents(self, strings):
        """
        Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

            return [self.tokenize(s) for s in strings]

        :rtype: list(list(str))
        """
        return [self.tokenize(s) for s in strings]

    def span_tokenize_sents(self, strings):
        """
        Apply ``self.span_tokenize()`` to each element of ``strings``.  I.e.:

            return [self.span_tokenize(s) for s in strings]

        :rtype: iter(list(tuple(int, int)))
        """
        for s in strings:
            yield list(self.span_tokenize(s))
