Ù­.
­
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
³
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨

StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-0-g3f878cff5b68Â-

embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*'
shared_nameembedding_1/embeddings

*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings* 
_output_shapes
:

*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
d*
dtype0
r
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
k
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes

:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm/lstm_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*(
shared_namelstm/lstm_cell_4/kernel

+lstm/lstm_cell_4/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell_4/kernel*
_output_shapes
:	
*
dtype0

!lstm/lstm_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*2
shared_name#!lstm/lstm_cell_4/recurrent_kernel

5lstm/lstm_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp!lstm/lstm_cell_4/recurrent_kernel*
_output_shapes
:	d*
dtype0

lstm/lstm_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namelstm/lstm_cell_4/bias
|
)lstm/lstm_cell_4/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell_4/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*.
shared_nameAdam/embedding_1/embeddings/m

1Adam/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/m* 
_output_shapes
:

*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
d*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
y
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes

:*
dtype0

Adam/lstm/lstm_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*/
shared_name Adam/lstm/lstm_cell_4/kernel/m

2Adam/lstm/lstm_cell_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell_4/kernel/m*
_output_shapes
:	
*
dtype0
­
(Adam/lstm/lstm_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*9
shared_name*(Adam/lstm/lstm_cell_4/recurrent_kernel/m
¦
<Adam/lstm/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/lstm/lstm_cell_4/recurrent_kernel/m*
_output_shapes
:	d*
dtype0

Adam/lstm/lstm_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/lstm/lstm_cell_4/bias/m

0Adam/lstm/lstm_cell_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell_4/bias/m*
_output_shapes	
:*
dtype0

Adam/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*.
shared_nameAdam/embedding_1/embeddings/v

1Adam/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/v* 
_output_shapes
:

*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
d*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
y
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes

:*
dtype0

Adam/lstm/lstm_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*/
shared_name Adam/lstm/lstm_cell_4/kernel/v

2Adam/lstm/lstm_cell_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell_4/kernel/v*
_output_shapes
:	
*
dtype0
­
(Adam/lstm/lstm_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*9
shared_name*(Adam/lstm/lstm_cell_4/recurrent_kernel/v
¦
<Adam/lstm/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/lstm/lstm_cell_4/recurrent_kernel/v*
_output_shapes
:	d*
dtype0

Adam/lstm/lstm_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/lstm/lstm_cell_4/bias/v

0Adam/lstm/lstm_cell_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell_4/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
ñ3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¬3
value¢3B3 B3
Î
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
 

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
¥
	variables
trainable_variables
 regularization_losses
!	keras_api
"_random_generator
#__call__
*$&call_and_return_all_conditional_losses* 
¦

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*
°
-iter

.beta_1

/beta_2
	0decay
1learning_ratemb%mc&md2me3mf4mgvh%vi&vj2vk3vl4vm*
.
0
21
32
43
%4
&5*
.
0
21
32
43
%4
&5*
* 
°
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

:serving_default* 
jd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 

;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
ã
@
state_size

2kernel
3recurrent_kernel
4bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E_random_generator
F__call__
*G&call_and_return_all_conditional_losses*
* 

20
31
42*

20
31
42*
* 


Hstates
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
 regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1*

%0
&1*
* 

Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm/lstm_cell_4/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!lstm/lstm_cell_4/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUElstm/lstm_cell_4/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

X0*
* 
* 
* 
* 
* 
* 
* 
* 
* 

20
31
42*

20
31
42*
* 

Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	^total
	_count
`	variables
a	keras_api*
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

^0
_1*

`	variables*

VARIABLE_VALUEAdam/embedding_1/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm/lstm_cell_4/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/lstm/lstm_cell_4/recurrent_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/lstm/lstm_cell_4/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/embedding_1/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm/lstm_cell_4/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/lstm/lstm_cell_4/recurrent_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/lstm/lstm_cell_4/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

!serving_default_embedding_1_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ô
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_1_inputembedding_1/embeddingslstm/lstm_cell_4/kernel!lstm/lstm_cell_4/recurrent_kernellstm/lstm_cell_4/biasdense_1/kerneldense_1/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1488194
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ë

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_1/embeddings/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+lstm/lstm_cell_4/kernel/Read/ReadVariableOp5lstm/lstm_cell_4/recurrent_kernel/Read/ReadVariableOp)lstm/lstm_cell_4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/embedding_1/embeddings/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp2Adam/lstm/lstm_cell_4/kernel/m/Read/ReadVariableOp<Adam/lstm/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOp0Adam/lstm/lstm_cell_4/bias/m/Read/ReadVariableOp1Adam/embedding_1/embeddings/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp2Adam/lstm/lstm_cell_4/kernel/v/Read/ReadVariableOp<Adam/lstm/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOp0Adam/lstm/lstm_cell_4/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_1490108
ò
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_1/embeddingsdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm/lstm_cell_4/kernel!lstm/lstm_cell_4/recurrent_kernellstm/lstm_cell_4/biastotalcountAdam/embedding_1/embeddings/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/lstm/lstm_cell_4/kernel/m(Adam/lstm/lstm_cell_4/recurrent_kernel/mAdam/lstm/lstm_cell_4/bias/mAdam/embedding_1/embeddings/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/lstm/lstm_cell_4/kernel/v(Adam/lstm/lstm_cell_4/recurrent_kernel/vAdam/lstm/lstm_cell_4/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_1490193,
Ê

)__inference_dense_1_layer_call_fn_1489999

inputs
unknown:
d
	unknown_0:

identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1486623q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

µ
&__inference_lstm_layer_call_fn_1488233
inputs_0
unknown:	

	unknown_0:	d
	unknown_1:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1486141o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0
íÂ
õ
=__inference___backward_gpu_lstm_with_fallback_1488931_1489107
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:ª
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:À
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ù
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Å
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ni
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:di
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:dø
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ì
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Nï
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dï
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:dm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   ¡
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:d
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:·
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:·
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:·
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:·
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:·
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:·
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:·
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
: ¯
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
µ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ê
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ò
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	dd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapesò
ï:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::::::: : : : *=
api_implements+)lstm_1ab015bf-c9c7-4ddb-8245-c01e3bd4bd3b*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1489106*
go_backwards( *

time_major( :- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd::6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: ::6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: 

_output_shapes
::1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:	

_output_shapes
:::
6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:"

_output_shapes

:Þ: 

_output_shapes
::-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
øÁ
õ
=__inference___backward_gpu_lstm_with_fallback_1486940_1487116
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:¡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:·
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdù
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ð
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Å
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ni
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:di
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:dø
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ì
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Nï
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dï
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:dm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   ¡
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:d
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:·
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:·
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:·
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:·
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:·
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:·
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:·
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
: ¯
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
µ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ê
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ò
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	dd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*é
_input_shapes×
Ô:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::::::: : : : *=
api_implements+)lstm_a323077f-9ed6-453f-a8f9-bfabec198d85*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1487115*
go_backwards( *

time_major( :- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd: 

_output_shapes
::1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:	

_output_shapes
::1
-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:"

_output_shapes

:Þ: 

_output_shapes
::-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ù:
Á
!__inference_standard_lstm_1488409

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1488324*
condR
while_cond_1488323*b
output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_766f7c66-41bb-4014-ae8f-8f4701abce90*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ò(
Ð
while_body_1487361
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :È
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :éèÒ_
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
:%	!

_output_shapes
:	d:!


_output_shapes	
:
·
½
A__inference_lstm_layer_call_and_return_conditional_losses_1488682
inputs_0/
read_readvariableop_resource:	
1
read_1_readvariableop_resource:	d-
read_2_readvariableop_resource:	

identity_3¢Read/ReadVariableOp¢Read_1/ReadVariableOp¢Read_2/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	dq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:¾
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_standard_lstm_1488409i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0
íÂ
õ
=__inference___backward_gpu_lstm_with_fallback_1488504_1488680
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:ª
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:À
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ù
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Å
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ni
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:di
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:dø
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ì
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Nï
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dï
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:dm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   ¡
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:d
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:·
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:·
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:·
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:·
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:·
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:·
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:·
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
: ¯
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
µ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ê
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ò
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	dd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapesò
ï:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::::::: : : : *=
api_implements+)lstm_766f7c66-41bb-4014-ae8f-8f4701abce90*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1488679*
go_backwards( *

time_major( :- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd::6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: ::6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: 

_output_shapes
::1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:	

_output_shapes
:::
6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:"

_output_shapes

:Þ: 

_output_shapes
::-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ò(
Ð
while_body_1487802
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :È
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :éèÒ_
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
:%	!

_output_shapes
:	d:!


_output_shapes	
:
¹L
¥
(__forward_gpu_lstm_with_fallback_1489106

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_1ab015bf-c9c7-4ddb-8245-c01e3bd4bd3b*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1488931_1489107*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ò(
Ð
while_body_1484906
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :È
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :éèÒ_
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
:%	!

_output_shapes
:	d:!


_output_shapes	
:
Ò(
Ð
while_body_1488324
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :È
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :éèÒ_
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
:%	!

_output_shapes
:	d:!


_output_shapes	
:
	
Æ
while_cond_1487360
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1487360___redundant_placeholder05
1while_while_cond_1487360___redundant_placeholder15
1while_while_cond_1487360___redundant_placeholder25
1while_while_cond_1487360___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
á@
Í
*__inference_gpu_lstm_with_fallback_1485524

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ÞÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_66b664ae-e547-4b2f-8a93-6f7b25cb4e82*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
£:
Á
!__inference_standard_lstm_1489263

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1489178*
condR
while_cond_1489177*b
output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_0eb21135-6683-4264-a3ab-65785a526e43*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ª	
¨
H__inference_embedding_1_layer_call_and_return_conditional_losses_1486167

inputs,
embedding_lookup_1486161:


identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
embedding_lookupResourceGatherembedding_lookup_1486161Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/1486161*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/1486161*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1486610

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
·
½
A__inference_lstm_layer_call_and_return_conditional_losses_1489109
inputs_0/
read_readvariableop_resource:	
1
read_1_readvariableop_resource:	d-
read_2_readvariableop_resource:	

identity_3¢Read/ReadVariableOp¢Read_1/ReadVariableOp¢Read_2/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	dq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:¾
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_standard_lstm_1488836i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0
ó
d
+__inference_dropout_1_layer_call_fn_1489973

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1486675o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
øÁ
õ
=__inference___backward_gpu_lstm_with_fallback_1485086_1485262
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:¡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:·
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdù
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ð
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Å
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ni
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:di
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:dø
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ì
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Nï
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dï
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:dm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   ¡
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:d
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:·
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:·
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:·
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:·
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:·
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:·
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:·
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
: ¯
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
µ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ê
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ò
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	dd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*é
_input_shapes×
Ô:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::::::: : : : *=
api_implements+)lstm_c3366eeb-bc60-49b0-8cce-2123674340fe*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1485261*
go_backwards( *

time_major( :- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd: 

_output_shapes
::1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:	

_output_shapes
::1
-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:"

_output_shapes

:Þ: 

_output_shapes
::-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ë4
ì
G__inference_sequential_layer_call_and_return_conditional_losses_1488175

inputs8
$embedding_1_embedding_lookup_1487731:

4
!lstm_read_readvariableop_resource:	
6
#lstm_read_1_readvariableop_resource:	d2
#lstm_read_2_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
d7
'dense_1_biasadd_readvariableop_resource:

identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢embedding_1/embedding_lookup¢lstm/Read/ReadVariableOp¢lstm/Read_1/ReadVariableOp¢lstm/Read_2/ReadVariableOpa
embedding_1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_1487731embedding_1/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding_1/embedding_lookup/1487731*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0Ç
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/1487731*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j

lstm/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
lstm/Read/ReadVariableOpReadVariableOp!lstm_read_readvariableop_resource*
_output_shapes
:	
*
dtype0e
lstm/IdentityIdentity lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	

lstm/Read_1/ReadVariableOpReadVariableOp#lstm_read_1_readvariableop_resource*
_output_shapes
:	d*
dtype0i
lstm/Identity_1Identity"lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	d{
lstm/Read_2/ReadVariableOpReadVariableOp#lstm_read_2_readvariableop_resource*
_output_shapes	
:*
dtype0e
lstm/Identity_2Identity"lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:û
lstm/PartitionedCallPartitionedCall0embedding_1/embedding_lookup/Identity_1:output:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/Identity:output:0lstm/Identity_1:output:0lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_standard_lstm_1487887\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_1/dropout/MulMullstm/PartitionedCall:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
dropout_1/dropout/ShapeShapelstm/PartitionedCall:output:0*
T0*
_output_shapes
: 
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ä
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes

:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding_1/embedding_lookup^lstm/Read/ReadVariableOp^lstm/Read_1/ReadVariableOp^lstm/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup24
lstm/Read/ReadVariableOplstm/Read/ReadVariableOp28
lstm/Read_1/ReadVariableOplstm/Read_1/ReadVariableOp28
lstm/Read_2/ReadVariableOplstm/Read_2/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô	
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1486675

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¢	

,__inference_sequential_layer_call_fn_1486645
embedding_1_input
unknown:


	unknown_0:	

	unknown_1:	d
	unknown_2:	
	unknown_3:
d
	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_1486630q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameembedding_1_input
þ
¦
G__inference_sequential_layer_call_and_return_conditional_losses_1487226
embedding_1_input'
embedding_1_1487209:


lstm_1487212:	

lstm_1487214:	d
lstm_1487216:	#
dense_1_1487220:
d
dense_1_1487222:

identity¢dense_1/StatefulPartitionedCall¢#embedding_1/StatefulPartitionedCall¢lstm/StatefulPartitionedCall÷
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_1487209*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_1_layer_call_and_return_conditional_losses_1486167
lstm/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0lstm_1487212lstm_1487214lstm_1487216*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1486597Ú
dropout_1/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1486610
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_1487220dense_1_1487222*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1486623y
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:Z V
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameembedding_1_input
Ò(
Ð
while_body_1485345
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :È
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :éèÒ_
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
:%	!

_output_shapes
:	d:!


_output_shapes	
:
øÁ
õ
=__inference___backward_gpu_lstm_with_fallback_1489358_1489534
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:¡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:·
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdù
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ð
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Å
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ni
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:di
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:dø
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ì
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Nï
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dï
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:dm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   ¡
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:d
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:·
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:·
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:·
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:·
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:·
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:·
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:·
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
: ¯
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
µ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ê
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ò
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	dd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*é
_input_shapes×
Ô:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::::::: : : : *=
api_implements+)lstm_0eb21135-6683-4264-a3ab-65785a526e43*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1489533*
go_backwards( *

time_major( :- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd: 

_output_shapes
::1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:	

_output_shapes
::1
-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:"

_output_shapes

:Þ: 

_output_shapes
::-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
L
¥
(__forward_gpu_lstm_with_fallback_1487115

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Î
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_a323077f-9ed6-453f-a8f9-bfabec198d85*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1486940_1487116*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ý:
¶
 __inference__traced_save_1490108
file_prefix5
1savev2_embedding_1_embeddings_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_lstm_lstm_cell_4_kernel_read_readvariableop@
<savev2_lstm_lstm_cell_4_recurrent_kernel_read_readvariableop4
0savev2_lstm_lstm_cell_4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_embedding_1_embeddings_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop=
9savev2_adam_lstm_lstm_cell_4_kernel_m_read_readvariableopG
Csavev2_adam_lstm_lstm_cell_4_recurrent_kernel_m_read_readvariableop;
7savev2_adam_lstm_lstm_cell_4_bias_m_read_readvariableop<
8savev2_adam_embedding_1_embeddings_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop=
9savev2_adam_lstm_lstm_cell_4_kernel_v_read_readvariableopG
Csavev2_adam_lstm_lstm_cell_4_recurrent_kernel_v_read_readvariableop;
7savev2_adam_lstm_lstm_cell_4_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*º
value°B­B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¡
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B ®
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_1_embeddings_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_lstm_lstm_cell_4_kernel_read_readvariableop<savev2_lstm_lstm_cell_4_recurrent_kernel_read_readvariableop0savev2_lstm_lstm_cell_4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_embedding_1_embeddings_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop9savev2_adam_lstm_lstm_cell_4_kernel_m_read_readvariableopCsavev2_adam_lstm_lstm_cell_4_recurrent_kernel_m_read_readvariableop7savev2_adam_lstm_lstm_cell_4_bias_m_read_readvariableop8savev2_adam_embedding_1_embeddings_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop9savev2_adam_lstm_lstm_cell_4_kernel_v_read_readvariableopCsavev2_adam_lstm_lstm_cell_4_recurrent_kernel_v_read_readvariableop7savev2_adam_lstm_lstm_cell_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Þ
_input_shapesÌ
É: :

:
d:: : : : : :	
:	d:: : :

:
d::	
:	d::

:
d::	
:	d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:

:&"
 
_output_shapes
:
d:"

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%	!

_output_shapes
:	
:%
!

_output_shapes
:	d:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:

:&"
 
_output_shapes
:
d:"

_output_shapes

::%!

_output_shapes
:	
:%!

_output_shapes
:	d:!

_output_shapes	
::&"
 
_output_shapes
:

:&"
 
_output_shapes
:
d:"

_output_shapes

::%!

_output_shapes
:	
:%!

_output_shapes
:	d:!

_output_shapes	
::

_output_shapes
: 
øÁ
õ
=__inference___backward_gpu_lstm_with_fallback_1486419_1486595
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:¡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:·
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdù
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ð
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Å
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ni
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:di
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:dø
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ì
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Nï
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dï
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:dm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   ¡
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:d
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:·
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:·
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:·
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:·
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:·
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:·
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:·
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
: ¯
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
µ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ê
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ò
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	dd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*é
_input_shapes×
Ô:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::::::: : : : *=
api_implements+)lstm_3f69c81b-ceb7-4a49-9aa1-bf18fca4c4fb*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1486594*
go_backwards( *

time_major( :- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd: 

_output_shapes
::1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:	

_output_shapes
::1
-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:"

_output_shapes

:Þ: 

_output_shapes
::-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

»
A__inference_lstm_layer_call_and_return_conditional_losses_1489536

inputs/
read_readvariableop_resource:	
1
read_1_readvariableop_resource:	d-
read_2_readvariableop_resource:	

identity_3¢Read/ReadVariableOp¢Read_1/ReadVariableOp¢Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	dq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:³
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_standard_lstm_1489263i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¦
Ê
G__inference_sequential_layer_call_and_return_conditional_losses_1487246
embedding_1_input'
embedding_1_1487229:


lstm_1487232:	

lstm_1487234:	d
lstm_1487236:	#
dense_1_1487240:
d
dense_1_1487242:

identity¢dense_1/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢#embedding_1/StatefulPartitionedCall¢lstm/StatefulPartitionedCall÷
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_1487229*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_1_layer_call_and_return_conditional_losses_1486167
lstm/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0lstm_1487232lstm_1487234lstm_1487236*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1487118ê
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1486675
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_1487240dense_1_1487242*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1486623y
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:Z V
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameembedding_1_input
á@
Í
*__inference_gpu_lstm_with_fallback_1488930

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ÞÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_1ab015bf-c9c7-4ddb-8245-c01e3bd4bd3b*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
¬

-__inference_embedding_1_layer_call_fn_1488201

inputs
unknown:


identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_1_layer_call_and_return_conditional_losses_1486167s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Æ
while_cond_1487801
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1487801___redundant_placeholder05
1while_while_cond_1487801___redundant_placeholder15
1while_while_cond_1487801___redundant_placeholder25
1while_while_cond_1487801___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ò(
Ð
while_body_1489605
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :È
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :éèÒ_
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
:%	!

_output_shapes
:	d:!


_output_shapes	
:
¯
»
A__inference_lstm_layer_call_and_return_conditional_losses_1486141

inputs/
read_readvariableop_resource:	
1
read_1_readvariableop_resource:	d-
read_2_readvariableop_resource:	

identity_3¢Read/ReadVariableOp¢Read_1/ReadVariableOp¢Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	dq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:¼
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_standard_lstm_1485868i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

»
A__inference_lstm_layer_call_and_return_conditional_losses_1489963

inputs/
read_readvariableop_resource:	
1
read_1_readvariableop_resource:	d-
read_2_readvariableop_resource:	

identity_3¢Read/ReadVariableOp¢Read_1/ReadVariableOp¢Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	dq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:³
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_standard_lstm_1489690i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
£:
Á
!__inference_standard_lstm_1487446

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1487361*
condR
while_cond_1487360*b
output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_99b0fdcc-5522-4a24-884f-3d364a385263*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
	
Æ
while_cond_1485782
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1485782___redundant_placeholder05
1while_while_cond_1485782___redundant_placeholder15
1while_while_cond_1485782___redundant_placeholder25
1while_while_cond_1485782___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ù:
Á
!__inference_standard_lstm_1485430

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1485345*
condR
while_cond_1485344*b
output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_66b664ae-e547-4b2f-8a93-6f7b25cb4e82*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
	

,__inference_sequential_layer_call_fn_1487269

inputs
unknown:


	unknown_0:	

	unknown_1:	d
	unknown_2:	
	unknown_3:
d
	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_1486630q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æf

#__inference__traced_restore_1490193
file_prefix;
'assignvariableop_embedding_1_embeddings:

5
!assignvariableop_1_dense_1_kernel:
d/
assignvariableop_2_dense_1_bias:
&
assignvariableop_3_adam_iter:	 (
assignvariableop_4_adam_beta_1: (
assignvariableop_5_adam_beta_2: '
assignvariableop_6_adam_decay: /
%assignvariableop_7_adam_learning_rate: =
*assignvariableop_8_lstm_lstm_cell_4_kernel:	
G
4assignvariableop_9_lstm_lstm_cell_4_recurrent_kernel:	d8
)assignvariableop_10_lstm_lstm_cell_4_bias:	#
assignvariableop_11_total: #
assignvariableop_12_count: E
1assignvariableop_13_adam_embedding_1_embeddings_m:

=
)assignvariableop_14_adam_dense_1_kernel_m:
d7
'assignvariableop_15_adam_dense_1_bias_m:
E
2assignvariableop_16_adam_lstm_lstm_cell_4_kernel_m:	
O
<assignvariableop_17_adam_lstm_lstm_cell_4_recurrent_kernel_m:	d?
0assignvariableop_18_adam_lstm_lstm_cell_4_bias_m:	E
1assignvariableop_19_adam_embedding_1_embeddings_v:

=
)assignvariableop_20_adam_dense_1_kernel_v:
d7
'assignvariableop_21_adam_dense_1_bias_v:
E
2assignvariableop_22_adam_lstm_lstm_cell_4_kernel_v:	
O
<assignvariableop_23_adam_lstm_lstm_cell_4_recurrent_kernel_v:	d?
0assignvariableop_24_adam_lstm_lstm_cell_4_bias_v:	
identity_26¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*º
value°B­B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B  
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_1_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp*assignvariableop_8_lstm_lstm_cell_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_9AssignVariableOp4assignvariableop_9_lstm_lstm_cell_4_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp)assignvariableop_10_lstm_lstm_cell_4_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_13AssignVariableOp1assignvariableop_13_adam_embedding_1_embeddings_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_dense_1_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_1_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_16AssignVariableOp2assignvariableop_16_adam_lstm_lstm_cell_4_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_17AssignVariableOp<assignvariableop_17_adam_lstm_lstm_cell_4_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_lstm_lstm_cell_4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_19AssignVariableOp1assignvariableop_19_adam_embedding_1_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_1_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_1_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_lstm_lstm_cell_4_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_23AssignVariableOp<assignvariableop_23_adam_lstm_lstm_cell_4_recurrent_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_lstm_lstm_cell_4_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 õ
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: â
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ª@
Í
*__inference_gpu_lstm_with_fallback_1487540

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ÞÊ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_99b0fdcc-5522-4a24-884f-3d364a385263*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
íÂ
õ
=__inference___backward_gpu_lstm_with_fallback_1485963_1486139
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:ª
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:À
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ù
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Å
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ni
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:di
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:dø
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ì
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Nï
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dï
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:dm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   ¡
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:d
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:·
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:·
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:·
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:·
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:·
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:·
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:·
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
: ¯
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
µ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ê
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ò
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	dd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapesò
ï:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::::::: : : : *=
api_implements+)lstm_fa7aebf8-d8e9-4db2-80f9-590434b8a9a5*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1486138*
go_backwards( *

time_major( :- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd::6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: ::6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: 

_output_shapes
::1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:	

_output_shapes
:::
6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:"

_output_shapes

:Þ: 

_output_shapes
::-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¹L
¥
(__forward_gpu_lstm_with_fallback_1488679

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_766f7c66-41bb-4014-ae8f-8f4701abce90*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1488504_1488680*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
á@
Í
*__inference_gpu_lstm_with_fallback_1485962

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ÞÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_fa7aebf8-d8e9-4db2-80f9-590434b8a9a5*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ö

%__inference_signature_wrapper_1488194
embedding_1_input
unknown:


	unknown_0:	

	unknown_1:	d
	unknown_2:	
	unknown_3:
d
	unknown_4:

identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1485272q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameembedding_1_input
ª@
Í
*__inference_gpu_lstm_with_fallback_1485085

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ÞÊ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_c3366eeb-bc60-49b0-8cce-2123674340fe*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ò(
Ð
while_body_1486760
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :È
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :éèÒ_
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
:%	!

_output_shapes
:	d:!


_output_shapes	
:

»
A__inference_lstm_layer_call_and_return_conditional_losses_1486597

inputs/
read_readvariableop_resource:	
1
read_1_readvariableop_resource:	d-
read_2_readvariableop_resource:	

identity_3¢Read/ReadVariableOp¢Read_1/ReadVariableOp¢Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	dq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:³
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_standard_lstm_1486324i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
	
Æ
while_cond_1488750
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1488750___redundant_placeholder05
1while_while_cond_1488750___redundant_placeholder15
1while_while_cond_1488750___redundant_placeholder25
1while_while_cond_1488750___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
L
¥
(__forward_gpu_lstm_with_fallback_1489960

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Î
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_7a2a5956-0623-43bb-b79b-e15af8f05463*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1489785_1489961*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ª@
Í
*__inference_gpu_lstm_with_fallback_1487981

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ÞÊ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_d63a27f7-1e0e-4d67-899b-824c4877d9a7*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ô	
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1489990

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ª	
¨
H__inference_embedding_1_layer_call_and_return_conditional_losses_1488211

inputs,
embedding_lookup_1488205:


identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
embedding_lookupResourceGatherembedding_lookup_1488205Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/1488205*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0£
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/1488205*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù:
Á
!__inference_standard_lstm_1485868

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1485783*
condR
while_cond_1485782*b
output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_fa7aebf8-d8e9-4db2-80f9-590434b8a9a5*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ò(
Ð
while_body_1486239
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :È
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :éèÒ_
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
:%	!

_output_shapes
:	d:!


_output_shapes	
:
£:
Á
!__inference_standard_lstm_1487887

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1487802*
condR
while_cond_1487801*b
output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_d63a27f7-1e0e-4d67-899b-824c4877d9a7*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
øÁ
õ
=__inference___backward_gpu_lstm_with_fallback_1489785_1489961
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:¡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:·
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdù
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ð
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Å
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ni
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:di
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:dø
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ì
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Nï
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dï
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:dm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   ¡
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:d
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:·
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:·
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:·
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:·
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:·
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:·
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:·
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
: ¯
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
µ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ê
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ò
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	dd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*é
_input_shapes×
Ô:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::::::: : : : *=
api_implements+)lstm_7a2a5956-0623-43bb-b79b-e15af8f05463*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1489960*
go_backwards( *

time_major( :- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd: 

_output_shapes
::1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:	

_output_shapes
::1
-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:"

_output_shapes

:Þ: 

_output_shapes
::-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ò(
Ð
while_body_1488751
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :È
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :éèÒ_
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
:%	!

_output_shapes
:	d:!


_output_shapes	
:
	
Æ
while_cond_1485344
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1485344___redundant_placeholder05
1while_while_cond_1485344___redundant_placeholder15
1while_while_cond_1485344___redundant_placeholder25
1while_while_cond_1485344___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
-
ì
G__inference_sequential_layer_call_and_return_conditional_losses_1487727

inputs8
$embedding_1_embedding_lookup_1487290:

4
!lstm_read_readvariableop_resource:	
6
#lstm_read_1_readvariableop_resource:	d2
#lstm_read_2_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
d7
'dense_1_biasadd_readvariableop_resource:

identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢embedding_1/embedding_lookup¢lstm/Read/ReadVariableOp¢lstm/Read_1/ReadVariableOp¢lstm/Read_2/ReadVariableOpa
embedding_1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_1487290embedding_1/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding_1/embedding_lookup/1487290*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0Ç
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/1487290*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j

lstm/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
lstm/Read/ReadVariableOpReadVariableOp!lstm_read_readvariableop_resource*
_output_shapes
:	
*
dtype0e
lstm/IdentityIdentity lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	

lstm/Read_1/ReadVariableOpReadVariableOp#lstm_read_1_readvariableop_resource*
_output_shapes
:	d*
dtype0i
lstm/Identity_1Identity"lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	d{
lstm/Read_2/ReadVariableOpReadVariableOp#lstm_read_2_readvariableop_resource*
_output_shapes	
:*
dtype0e
lstm/Identity_2Identity"lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:û
lstm/PartitionedCallPartitionedCall0embedding_1/embedding_lookup/Identity_1:output:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/Identity:output:0lstm/Identity_1:output:0lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_standard_lstm_1487446o
dropout_1/IdentityIdentitylstm/PartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes

:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding_1/embedding_lookup^lstm/Read/ReadVariableOp^lstm/Read_1/ReadVariableOp^lstm/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup24
lstm/Read/ReadVariableOplstm/Read/ReadVariableOp28
lstm/Read_1/ReadVariableOplstm/Read_1/ReadVariableOp28
lstm/Read_2/ReadVariableOplstm/Read_2/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£:
Á
!__inference_standard_lstm_1486845

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1486760*
condR
while_cond_1486759*b
output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_a323077f-9ed6-453f-a8f9-bfabec198d85*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ò(
Ð
while_body_1485783
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :È
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :éèÒ_
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
:%	!

_output_shapes
:	d:!


_output_shapes	
:
£:
Á
!__inference_standard_lstm_1489690

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1489605*
condR
while_cond_1489604*b
output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_7a2a5956-0623-43bb-b79b-e15af8f05463*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ª@
Í
*__inference_gpu_lstm_with_fallback_1489357

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ÞÊ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_0eb21135-6683-4264-a3ab-65785a526e43*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ª@
Í
*__inference_gpu_lstm_with_fallback_1486939

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ÞÊ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_a323077f-9ed6-453f-a8f9-bfabec198d85*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ý

G__inference_sequential_layer_call_and_return_conditional_losses_1486630

inputs'
embedding_1_1486168:


lstm_1486598:	

lstm_1486600:	d
lstm_1486602:	#
dense_1_1486624:
d
dense_1_1486626:

identity¢dense_1/StatefulPartitionedCall¢#embedding_1/StatefulPartitionedCall¢lstm/StatefulPartitionedCallì
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_1486168*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_1_layer_call_and_return_conditional_losses_1486167
lstm/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0lstm_1486598lstm_1486600lstm_1486602*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1486597Ú
dropout_1/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1486610
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_1486624dense_1_1486626*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1486623y
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó5
Ö
"__inference__wrapped_model_1485272
embedding_1_inputC
/sequential_embedding_1_embedding_lookup_1484835:

?
,sequential_lstm_read_readvariableop_resource:	
A
.sequential_lstm_read_1_readvariableop_resource:	d=
.sequential_lstm_read_2_readvariableop_resource:	E
1sequential_dense_1_matmul_readvariableop_resource:
dB
2sequential_dense_1_biasadd_readvariableop_resource:

identity¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢'sequential/embedding_1/embedding_lookup¢#sequential/lstm/Read/ReadVariableOp¢%sequential/lstm/Read_1/ReadVariableOp¢%sequential/lstm/Read_2/ReadVariableOpw
sequential/embedding_1/CastCastembedding_1_input*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/embedding_1/embedding_lookupResourceGather/sequential_embedding_1_embedding_lookup_1484835sequential/embedding_1/Cast:y:0*
Tindices0*B
_class8
64loc:@sequential/embedding_1/embedding_lookup/1484835*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0è
0sequential/embedding_1/embedding_lookup/IdentityIdentity0sequential/embedding_1/embedding_lookup:output:0*
T0*B
_class8
64loc:@sequential/embedding_1/embedding_lookup/1484835*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
2sequential/embedding_1/embedding_lookup/Identity_1Identity9sequential/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

sequential/lstm/ShapeShape;sequential/embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:m
#sequential/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%sequential/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%sequential/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d£
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
sequential/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d§
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
sequential/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¢
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#sequential/lstm/Read/ReadVariableOpReadVariableOp,sequential_lstm_read_readvariableop_resource*
_output_shapes
:	
*
dtype0{
sequential/lstm/IdentityIdentity+sequential/lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	

%sequential/lstm/Read_1/ReadVariableOpReadVariableOp.sequential_lstm_read_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
sequential/lstm/Identity_1Identity-sequential/lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	d
%sequential/lstm/Read_2/ReadVariableOpReadVariableOp.sequential_lstm_read_2_readvariableop_resource*
_output_shapes	
:*
dtype0{
sequential/lstm/Identity_2Identity-sequential/lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:È
sequential/lstm/PartitionedCallPartitionedCall;sequential/embedding_1/embedding_lookup/Identity_1:output:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0!sequential/lstm/Identity:output:0#sequential/lstm/Identity_1:output:0#sequential/lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_standard_lstm_1484991
sequential/dropout_1/IdentityIdentity(sequential/lstm/PartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0±
sequential/dense_1/MatMulMatMul&sequential/dropout_1/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes

:*
dtype0±
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
IdentityIdentity$sequential/dense_1/Softmax:softmax:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp(^sequential/embedding_1/embedding_lookup$^sequential/lstm/Read/ReadVariableOp&^sequential/lstm/Read_1/ReadVariableOp&^sequential/lstm/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2R
'sequential/embedding_1/embedding_lookup'sequential/embedding_1/embedding_lookup2J
#sequential/lstm/Read/ReadVariableOp#sequential/lstm/Read/ReadVariableOp2N
%sequential/lstm/Read_1/ReadVariableOp%sequential/lstm/Read_1/ReadVariableOp2N
%sequential/lstm/Read_2/ReadVariableOp%sequential/lstm/Read_2/ReadVariableOp:Z V
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameembedding_1_input

µ
&__inference_lstm_layer_call_fn_1488222
inputs_0
unknown:	

	unknown_0:	d
	unknown_1:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1485703o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0
¯
»
A__inference_lstm_layer_call_and_return_conditional_losses_1485703

inputs/
read_readvariableop_resource:	
1
read_1_readvariableop_resource:	d-
read_2_readvariableop_resource:	

identity_3¢Read/ReadVariableOp¢Read_1/ReadVariableOp¢Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	dq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:¼
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_standard_lstm_1485430i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¢	

,__inference_sequential_layer_call_fn_1487206
embedding_1_input
unknown:


	unknown_0:	

	unknown_1:	d
	unknown_2:	
	unknown_3:
d
	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_1487174q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameembedding_1_input
£:
Á
!__inference_standard_lstm_1486324

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1486239*
condR
while_cond_1486238*b
output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_3f69c81b-ceb7-4a49-9aa1-bf18fca4c4fb*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
øÁ
õ
=__inference___backward_gpu_lstm_with_fallback_1487982_1488158
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:¡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:·
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdù
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ð
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Å
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ni
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:di
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:dø
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ì
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Nï
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dï
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:dm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   ¡
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:d
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:·
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:·
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:·
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:·
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:·
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:·
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:·
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
: ¯
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
µ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ê
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ò
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	dd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*é
_input_shapes×
Ô:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::::::: : : : *=
api_implements+)lstm_d63a27f7-1e0e-4d67-899b-824c4877d9a7*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1488157*
go_backwards( *

time_major( :- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd: 

_output_shapes
::1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:	

_output_shapes
::1
-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:"

_output_shapes

:Þ: 

_output_shapes
::-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

»
A__inference_lstm_layer_call_and_return_conditional_losses_1487118

inputs/
read_readvariableop_resource:	
1
read_1_readvariableop_resource:	d-
read_2_readvariableop_resource:	

identity_3¢Read/ReadVariableOp¢Read_1/ReadVariableOp¢Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	dq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:³
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_standard_lstm_1486845i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ò(
Ð
while_body_1489178
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :È
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :éèÒ_
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
:%	!

_output_shapes
:	d:!


_output_shapes	
:
	
Æ
while_cond_1486238
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1486238___redundant_placeholder05
1while_while_cond_1486238___redundant_placeholder15
1while_while_cond_1486238___redundant_placeholder25
1while_while_cond_1486238___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ª@
Í
*__inference_gpu_lstm_with_fallback_1486418

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ÞÊ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_3f69c81b-ceb7-4a49-9aa1-bf18fca4c4fb*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ª@
Í
*__inference_gpu_lstm_with_fallback_1489784

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ÞÊ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_7a2a5956-0623-43bb-b79b-e15af8f05463*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ó
³
&__inference_lstm_layer_call_fn_1488244

inputs
unknown:	

	unknown_0:	d
	unknown_1:	
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1486597o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
	
Æ
while_cond_1484905
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1484905___redundant_placeholder05
1while_while_cond_1484905___redundant_placeholder15
1while_while_cond_1484905___redundant_placeholder25
1while_while_cond_1484905___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
¹L
¥
(__forward_gpu_lstm_with_fallback_1486138

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_fa7aebf8-d8e9-4db2-80f9-590434b8a9a5*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1485963_1486139*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
£:
Á
!__inference_standard_lstm_1484991

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1484906*
condR
while_cond_1484905*b
output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_c3366eeb-bc60-49b0-8cce-2123674340fe*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
L
¥
(__forward_gpu_lstm_with_fallback_1486594

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Î
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_3f69c81b-ceb7-4a49-9aa1-bf18fca4c4fb*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1486419_1486595*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
°

ù
D__inference_dense_1_layer_call_and_return_conditional_losses_1490010

inputs2
matmul_readvariableop_resource:
d/
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0k
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:*
dtype0x
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
SoftmaxSoftmaxBiasAdd:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
	
Æ
while_cond_1488323
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1488323___redundant_placeholder05
1while_while_cond_1488323___redundant_placeholder15
1while_while_cond_1488323___redundant_placeholder25
1while_while_cond_1488323___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
	

,__inference_sequential_layer_call_fn_1487286

inputs
unknown:


	unknown_0:	

	unknown_1:	d
	unknown_2:	
	unknown_3:
d
	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_1487174q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Æ
while_cond_1489604
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1489604___redundant_placeholder05
1while_while_cond_1489604___redundant_placeholder15
1while_while_cond_1489604___redundant_placeholder25
1while_while_cond_1489604___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
L
¥
(__forward_gpu_lstm_with_fallback_1485261

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Î
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_c3366eeb-bc60-49b0-8cce-2123674340fe*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1485086_1485262*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
	
Æ
while_cond_1486759
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1486759___redundant_placeholder05
1while_while_cond_1486759___redundant_placeholder15
1while_while_cond_1486759___redundant_placeholder25
1while_while_cond_1486759___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
°

ù
D__inference_dense_1_layer_call_and_return_conditional_losses_1486623

inputs2
matmul_readvariableop_resource:
d/
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0k
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:*
dtype0x
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
SoftmaxSoftmaxBiasAdd:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¹L
¥
(__forward_gpu_lstm_with_fallback_1485700

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_66b664ae-e547-4b2f-8a93-6f7b25cb4e82*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1485525_1485701*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
øÁ
õ
=__inference___backward_gpu_lstm_with_fallback_1487541_1487717
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:¡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:·
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdù
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ð
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Å
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ni
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:di
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:dø
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ì
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Nï
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dï
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:dm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   ¡
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:d
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:·
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:·
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:·
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:·
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:·
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:·
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:·
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
: ¯
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
µ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ê
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ò
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	dd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*é
_input_shapes×
Ô:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::::::: : : : *=
api_implements+)lstm_99b0fdcc-5522-4a24-884f-3d364a385263*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1487716*
go_backwards( *

time_major( :- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd: 

_output_shapes
::1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:	

_output_shapes
::1
-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:"

_output_shapes

:Þ: 

_output_shapes
::-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	
Æ
while_cond_1489177
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1489177___redundant_placeholder05
1while_while_cond_1489177___redundant_placeholder15
1while_while_cond_1489177___redundant_placeholder25
1while_while_cond_1489177___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ù:
Á
!__inference_standard_lstm_1488836

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1488751*
condR
while_cond_1488750*b
output_shapesQ
O: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : :	
:	d:*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_1ab015bf-c9c7-4ddb-8245-c01e3bd4bd3b*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ù
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1489978

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¡
G
+__inference_dropout_1_layer_call_fn_1489968

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1486610`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
L
¥
(__forward_gpu_lstm_with_fallback_1488157

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Î
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_d63a27f7-1e0e-4d67-899b-824c4877d9a7*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1487982_1488158*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
L
¥
(__forward_gpu_lstm_with_fallback_1489533

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Î
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_0eb21135-6683-4264-a3ab-65785a526e43*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1489358_1489534*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
L
¥
(__forward_gpu_lstm_with_fallback_1487716

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Î
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]

Identity_1Identitytranspose_9:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_99b0fdcc-5522-4a24-884f-3d364a385263*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1487541_1487717*
go_backwards( *

time_major( :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias

¿
G__inference_sequential_layer_call_and_return_conditional_losses_1487174

inputs'
embedding_1_1487157:


lstm_1487160:	

lstm_1487162:	d
lstm_1487164:	#
dense_1_1487168:
d
dense_1_1487170:

identity¢dense_1/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢#embedding_1/StatefulPartitionedCall¢lstm/StatefulPartitionedCallì
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_1487157*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_1_layer_call_and_return_conditional_losses_1486167
lstm/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0lstm_1487160lstm_1487162lstm_1487164*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1487118ê
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1486675
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_1487168dense_1_1487170*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1486623y
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
íÂ
õ
=__inference___backward_gpu_lstm_with_fallback_1485525_1485701
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:ª
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:À
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ù
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Å
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:èj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Nj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ni
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:di
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dj
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:dø
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ì
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:èð
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Nð
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Nï
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dï
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:dò
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:dm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   ¡
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   
   §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:d
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d£
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d¦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:d
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:·
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:·
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:·
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
d
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:·
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:·
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:·
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:·
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
: ¯
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
µ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ê
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ò
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	dd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapesò
ï:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd::ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:Þ::ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::::::: : : : *=
api_implements+)lstm_66b664ae-e547-4b2f-8a93-6f7b25cb4e82*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1485700*
go_backwards( *

time_major( :- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd::6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: ::6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: 

_output_shapes
::1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:	

_output_shapes
:::
6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:"

_output_shapes

:Þ: 

_output_shapes
::-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ó
³
&__inference_lstm_layer_call_fn_1488255

inputs
unknown:	

	unknown_0:	d
	unknown_1:	
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1487118o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
á@
Í
*__inference_gpu_lstm_with_fallback_1488503

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
d:
d:
d:
d*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:dd:dd:dd:dd*
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
: S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:d:d:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:d
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:d
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:èa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Na
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:N[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:d\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:d\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:dO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ÞÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:	
:	d:*=
api_implements+)lstm_766f7c66-41bb-4014-ae8f-8f4701abce90*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_h:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinit_c:GC

_output_shapes
:	

 
_user_specified_namekernel:QM

_output_shapes
:	d
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*À
serving_default¬
O
embedding_1_input:
#serving_default_embedding_1_input:0ÿÿÿÿÿÿÿÿÿ=
dense_12
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:îu
è
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
µ

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
¼
	variables
trainable_variables
 regularization_losses
!	keras_api
"_random_generator
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
»

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
¿
-iter

.beta_1

/beta_2
	0decay
1learning_ratemb%mc&md2me3mf4mgvh%vi&vj2vk3vl4vm"
	optimizer
J
0
21
32
43
%4
&5"
trackable_list_wrapper
J
0
21
32
43
%4
&5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
þ2û
,__inference_sequential_layer_call_fn_1486645
,__inference_sequential_layer_call_fn_1487269
,__inference_sequential_layer_call_fn_1487286
,__inference_sequential_layer_call_fn_1487206À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_layer_call_and_return_conditional_losses_1487727
G__inference_sequential_layer_call_and_return_conditional_losses_1488175
G__inference_sequential_layer_call_and_return_conditional_losses_1487226
G__inference_sequential_layer_call_and_return_conditional_losses_1487246À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
×BÔ
"__inference__wrapped_model_1485272embedding_1_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
:serving_default"
signature_map
*:(

2embedding_1/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_embedding_1_layer_call_fn_1488201¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_embedding_1_layer_call_and_return_conditional_losses_1488211¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø
@
state_size

2kernel
3recurrent_kernel
4bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E_random_generator
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Hstates
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
û2ø
&__inference_lstm_layer_call_fn_1488222
&__inference_lstm_layer_call_fn_1488233
&__inference_lstm_layer_call_fn_1488244
&__inference_lstm_layer_call_fn_1488255Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
A__inference_lstm_layer_call_and_return_conditional_losses_1488682
A__inference_lstm_layer_call_and_return_conditional_losses_1489109
A__inference_lstm_layer_call_and_return_conditional_losses_1489536
A__inference_lstm_layer_call_and_return_conditional_losses_1489963Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
 regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_1_layer_call_fn_1489968
+__inference_dropout_1_layer_call_fn_1489973´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_1_layer_call_and_return_conditional_losses_1489978
F__inference_dropout_1_layer_call_and_return_conditional_losses_1489990´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
": 
d2dense_1/kernel
:2dense_1/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_1_layer_call_fn_1489999¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_1_layer_call_and_return_conditional_losses_1490010¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(	
2lstm/lstm_cell_4/kernel
4:2	d2!lstm/lstm_cell_4/recurrent_kernel
$:"2lstm/lstm_cell_4/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
X0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÖBÓ
%__inference_signature_wrapper_1488194embedding_1_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ä2Á¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ä2Á¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	^total
	_count
`	variables
a	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
^0
_1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
/:-

2Adam/embedding_1/embeddings/m
':%
d2Adam/dense_1/kernel/m
!:2Adam/dense_1/bias/m
/:-	
2Adam/lstm/lstm_cell_4/kernel/m
9:7	d2(Adam/lstm/lstm_cell_4/recurrent_kernel/m
):'2Adam/lstm/lstm_cell_4/bias/m
/:-

2Adam/embedding_1/embeddings/v
':%
d2Adam/dense_1/kernel/v
!:2Adam/dense_1/bias/v
/:-	
2Adam/lstm/lstm_cell_4/kernel/v
9:7	d2(Adam/lstm/lstm_cell_4/recurrent_kernel/v
):'2Adam/lstm/lstm_cell_4/bias/v
"__inference__wrapped_model_1485272y234%&:¢7
0¢-
+(
embedding_1_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_1# 
dense_1ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense_1_layer_call_and_return_conditional_losses_1490010^%&/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_1_layer_call_fn_1489999Q%&/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dropout_1_layer_call_and_return_conditional_losses_1489978\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ¦
F__inference_dropout_1_layer_call_and_return_conditional_losses_1489990\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ~
+__inference_dropout_1_layer_call_fn_1489968O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "ÿÿÿÿÿÿÿÿÿd~
+__inference_dropout_1_layer_call_fn_1489973O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p
ª "ÿÿÿÿÿÿÿÿÿd«
H__inference_embedding_1_layer_call_and_return_conditional_losses_1488211_/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 
-__inference_embedding_1_layer_call_fn_1488201R/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
Â
A__inference_lstm_layer_call_and_return_conditional_losses_1488682}234O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ


 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 Â
A__inference_lstm_layer_call_and_return_conditional_losses_1489109}234O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ


 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ²
A__inference_lstm_layer_call_and_return_conditional_losses_1489536m234?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ²
A__inference_lstm_layer_call_and_return_conditional_losses_1489963m234?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
&__inference_lstm_layer_call_fn_1488222p234O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ


 
p 

 
ª "ÿÿÿÿÿÿÿÿÿd
&__inference_lstm_layer_call_fn_1488233p234O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ


 
p

 
ª "ÿÿÿÿÿÿÿÿÿd
&__inference_lstm_layer_call_fn_1488244`234?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p 

 
ª "ÿÿÿÿÿÿÿÿÿd
&__inference_lstm_layer_call_fn_1488255`234?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ


 
p

 
ª "ÿÿÿÿÿÿÿÿÿdÀ
G__inference_sequential_layer_call_and_return_conditional_losses_1487226u234%&B¢?
8¢5
+(
embedding_1_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 À
G__inference_sequential_layer_call_and_return_conditional_losses_1487246u234%&B¢?
8¢5
+(
embedding_1_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 µ
G__inference_sequential_layer_call_and_return_conditional_losses_1487727j234%&7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 µ
G__inference_sequential_layer_call_and_return_conditional_losses_1488175j234%&7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_sequential_layer_call_fn_1486645h234%&B¢?
8¢5
+(
embedding_1_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1487206h234%&B¢?
8¢5
+(
embedding_1_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1487269]234%&7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1487286]234%&7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¸
%__inference_signature_wrapper_1488194234%&O¢L
¢ 
EªB
@
embedding_1_input+(
embedding_1_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_1# 
dense_1ÿÿÿÿÿÿÿÿÿ