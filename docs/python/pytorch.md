# Pytorch

- `1xn` + `Kxn` is OK, more rules refer to [BROADCASTING SEMANTICS](https://pytorch.org/docs/master/notes/broadcasting.html#broadcasting-semantics), see also [:link:](https://github.com/szcf-weiya/MonotoneSplines.jl/blob/b22db54ba2afad3e52cc2512f64b5c816622cb4f/src/boot.py#L232-L234)
- save model, refer to [Recommended approach for saving a model](https://github.com/pytorch/pytorch/blob/761d6799beb3afa03657a71776412a2171ee7533/docs/source/notes/serialization.rst), see also [:link:](https://github.com/szcf-weiya/MonotoneSplines.jl/blob/b22db54ba2afad3e52cc2512f64b5c816622cb4f/src/boot.py#L262-L264)

```python
torch.save(the_model.state_dict(), PATH)
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
```