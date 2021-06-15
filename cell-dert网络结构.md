```shell
CellDETR(
  (backbone): Backbone(
    (input_convolution): Sequential(
      (0): Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (1): AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
    )
    (blocks): ModuleList(
      (0): ResNetBlock(
        (main_mapping): Sequential(
          (0): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01)
          (3): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): LeakyReLU(negative_slope=0.01)
        )
        (residual_mapping): Conv2d(1, 64, kernel_size=(1, 1), stride=(1, 1))
        (pooling): AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
      )
      (1): ResNetBlock(
        (main_mapping): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01)
          (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): LeakyReLU(negative_slope=0.01)
        )
        (residual_mapping): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        (pooling): AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
      )
      (2): ResNetBlock(
        (main_mapping): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01)
          (3): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): LeakyReLU(negative_slope=0.01)
        )
        (residual_mapping): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        (pooling): AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
      )
      (3): ResNetBlock(
        (main_mapping): Sequential(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01)
          (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): LeakyReLU(negative_slope=0.01)
        )
        (residual_mapping): Identity()
        (pooling): AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
      )
    )
  )
  (convolution_mapping): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
  (transformer): Transformer(
    (encoder): TransformerEncoder(
      (layers): ModuleList(
        (0): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): Linear(in_features=128, out_features=128, bias=True)
          )
          (linear1): Linear(in_features=128, out_features=512, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (linear2): Linear(in_features=512, out_features=128, bias=True)
          (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.0, inplace=False)
          (dropout2): Dropout(p=0.0, inplace=False)
          (activation): LeakyReLU(negative_slope=0.01)
        )
        (1): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): Linear(in_features=128, out_features=128, bias=True)
          )
          (linear1): Linear(in_features=128, out_features=512, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (linear2): Linear(in_features=512, out_features=128, bias=True)
          (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.0, inplace=False)
          (dropout2): Dropout(p=0.0, inplace=False)
          (activation): LeakyReLU(negative_slope=0.01)
        )
        (2): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): Linear(in_features=128, out_features=128, bias=True)
          )
          (linear1): Linear(in_features=128, out_features=512, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (linear2): Linear(in_features=512, out_features=128, bias=True)
          (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.0, inplace=False)
          (dropout2): Dropout(p=0.0, inplace=False)
          (activation): LeakyReLU(negative_slope=0.01)
        )
      )
    )
    (decoder): TransformerDecoder(
      (layers): ModuleList(
        (0): TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): Linear(in_features=128, out_features=128, bias=True)
          )
          (multihead_attn): MultiheadAttention(
            (out_proj): Linear(in_features=128, out_features=128, bias=True)
          )
          (linear1): Linear(in_features=128, out_features=512, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (linear2): Linear(in_features=512, out_features=128, bias=True)
          (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.0, inplace=False)
          (dropout2): Dropout(p=0.0, inplace=False)
          (dropout3): Dropout(p=0.0, inplace=False)
          (activation): LeakyReLU(negative_slope=0.01)
        )
        (1): TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): Linear(in_features=128, out_features=128, bias=True)
          )
          (multihead_attn): MultiheadAttention(
            (out_proj): Linear(in_features=128, out_features=128, bias=True)
          )
          (linear1): Linear(in_features=128, out_features=512, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (linear2): Linear(in_features=512, out_features=128, bias=True)
          (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.0, inplace=False)
          (dropout2): Dropout(p=0.0, inplace=False)
          (dropout3): Dropout(p=0.0, inplace=False)
          (activation): LeakyReLU(negative_slope=0.01)
        )
      )
      (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    )
  )
  (bounding_box_head): BoundingBoxHead(
    (layers): Sequential(
      (0): Linear(in_features=128, out_features=64, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): Linear(in_features=64, out_features=16, bias=True)
      (3): LeakyReLU(negative_slope=0.01)
      (4): Linear(in_features=16, out_features=4, bias=True)
    )
  )
  (class_head): Sequential(
    (0): Linear(in_features=128, out_features=64, bias=True)
    (1): LeakyReLU(negative_slope=0.01)
    (2): Linear(in_features=64, out_features=3, bias=True)
  )
  (segmentation_attention_head): MultiHeadAttention(
    (layer_box_embedding): Linear(in_features=128, out_features=128, bias=True)
    (layer_image_encoding): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
  )
  (segmentation_head): SegmentationHead(
    (blocks): ModuleList(
      (0): ResFeaturePyramidBlock(
        (main_mapping): Sequential(
          (0): Conv2d(136, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01)
          (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (4): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): LeakyReLU(negative_slope=0.01)
        )
        (residual_mapping): Conv2d(136, 64, kernel_size=(1, 1), stride=(1, 1))
        (upsampling): Upsample(scale_factor=(2.0, 2.0), mode=bicubic)
        (feature_mapping): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ResFeaturePyramidBlock(
        (main_mapping): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01)
          (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (4): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): LeakyReLU(negative_slope=0.01)
        )
        (residual_mapping): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
        (upsampling): Upsample(scale_factor=(2.0, 2.0), mode=bicubic)
        (feature_mapping): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): ResFeaturePyramidBlock(
        (main_mapping): Sequential(
          (0): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(negative_slope=0.01)
          (3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (4): InstanceNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): LeakyReLU(negative_slope=0.01)
        )
        (residual_mapping): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
        (upsampling): Upsample(scale_factor=(2.0, 2.0), mode=bicubic)
        (feature_mapping): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (final_block): FinalBlockReshaped(
      (main_mapping): Sequential(
        (0): Conv2d(384, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): InstanceNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01)
        (3): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): InstanceNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): LeakyReLU(negative_slope=0.01)
      )
      (residual_mapping): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
      (upsampling): Upsample(scale_factor=(2.0, 2.0), mode=bicubic)
      (final_mapping): Sequential(
        (0): Conv2d(48, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): LeakyReLU(negative_slope=0.01)
        (2): Conv2d(32, 12, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
  (segmentation_final_activation): Softmax(dim=1)
)

```

