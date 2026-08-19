Because the transpose makes the matrix dimensions line up and gives us the gradient for each weight.

Let's use exact operation:

```
Tmp [BX, F] × Wout [F, D] = Out [BX, D]
```


We want:

```
dWout [F, D]
```


During backward, we have:

```
dOut [BX, D]
```


The question is:
How do we combine Tmp and dOut to get dWout?

## 1. Look at the dimensions

We have:

```
Tmp          [BX, F]
dOut         [BX, D]

```

If we tried:

```
Tmp × dOut
```


we would have:

```
[BX, F] × [BX, D]
```


The middle dimensions are:

```
F ≠ BX
```


So this multiplication isn't even valid.

But if we transpose Tmp:

```
Tmp^T = [F, BX]
```


then:

```
Tmp^T × dOut

[F, BX] × [BX, D]

        ↓

[F, D]

```

And that's exactly the shape of Wout:

```
Wout = [F, D]
dWout = [F, D]
```


So the transpose is required.

## 2. But there's a deeper reason

Remember our simple scalar example:

```
Out = Tmp × Wout
```


For a single example, the gradient was:

```
dWout = Tmp × dOut
```


Now suppose we have multiple examples.

Say:

```
Tmp:

example 1 → [t11, t12]
example 2 → [t21, t22]
example 3 → [t31, t32]
```


So:

```
Tmp =
[
 t11 t12
 t21 t22
 t31 t32
]
```


Shape:

```
[3, 2]

```

And:

```
dOut =
[
 g11 g12
 g21 g22
 g31 g32
]
```


also [3, 2].

We need one gradient for each weight connecting the F input features to the D output features.

For example, the gradient for Wout[1,1] should be:

```
t11*g11 + t21*g21 + t31*g31
```


Notice what's happening:

we multiply corresponding examples and sum across the batch.

That's exactly what:

```
Tmp^T × dOut
```

does.


```
        examples
          ↓
Tmp^T = [ t11 t21 t31 ]
        [ t12 t22 t32 ]

dOut  = [ g11 g12 ]
        [ g21 g22 ]
        [ g31 g32 ]

              ↓

        [t11*g11 + t21*g21 + t31*g31, ...]
        [t12*g11 + t22*g21 + t32*g31, ...]
```


The batch dimension BX gets summed away.

That's why the transpose is so meaningful:

```
Tmp:       [BX, F]
             ↑
          examples

Tmp^T:     [F, BX]
             ↑
          features
```


Then:

```
Tmp^T × dOut
[F, BX] × [BX, D]
      ↓
    [F, D]

```

## 3. General rule

For a forward matrix multiplication:

A × W = Y


where:

```
A = [B, input_dim]
W = [input_dim, output_dim]
Y = [B, output_dim]
```

the weight gradient is:

```
dW = A^T × dY
```


because:

```
A^T       [input_dim, B]
dY        [B, output_dim]
                    ↓
dW        [input_dim, output_dim]
```


So in your case:

```
Forward:

Tmp [BX,F] × Wout [F,D] = Out [BX,D]


Backward:

Tmp^T [F,BX] × dOut [BX,D] = dWout [F,D]
```


**The transpose both makes the dimensions work and causes the gradients from all examples in the batch to accumulate into each weight.