// Diagnostic host-execution candidate. Keep the eager ATen operation graph;
// no custom CUDA arithmetic, recurrent rewrite, allocator pool or RNG calls.
#include <ATen/ATen.h>
#include <torch/library.h>

namespace {
at::Tensor final_state(
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& log_decay,
    const at::Tensor& beta,
    const std::optional<at::Tensor>& initial_state,
    int64_t chunk_size) {
  TORCH_CHECK(chunk_size > 0 && k.dim() == 4 && k.size(1) > 0,
              "KDA candidate needs a nonempty sequence and positive chunk");
  auto state = initial_state.has_value()
      ? *initial_state
      : at::zeros({k.size(0), k.size(2), k.size(3), v.size(3)},
                  k.options().dtype(at::kFloat));
  for (int64_t start = 0; start < k.size(1); start += chunk_size) {
    const auto end = std::min(start + chunk_size, k.size(1));
    const auto key = k.slice(1, start, end).to(at::kFloat).transpose(1, 2);
    const auto value = v.slice(1, start, end).to(at::kFloat).transpose(1, 2);
    const auto gates = log_decay.slice(1, start, end)
                           .to(at::kFloat).transpose(1, 2).cumsum(-2);
    const auto rate = beta.slice(1, start, end)
                          .to(at::kFloat).transpose(1, 2).unsqueeze(-1);
    const auto length = end - start;
    const auto invalid = at::ones({length, length}, k.options().dtype(at::kBool))
                             .tril().bitwise_not();
    const auto decay = (gates.unsqueeze(-2) - gates.unsqueeze(-3))
                           .masked_fill(invalid.unsqueeze(0).unsqueeze(0).unsqueeze(-1), 0)
                           .exp();
    const auto past_keys =
        (key.unsqueeze(-2) * key.unsqueeze(-3) * decay).sum(-1);
    auto triangular = (past_keys * rate).tril(-1);
    triangular = triangular + at::eye(length, k.options().dtype(at::kFloat));
    const auto old_read = at::matmul(key * gates.exp(), state);
    const auto writes = at::linalg_solve_triangular(
        triangular, rate * (value - old_read), false);
    const auto last = gates.slice(-2, length - 1, length);
    const auto final_keys = key * (last - gates).exp();
    state = gates.select(-2, length - 1).exp().unsqueeze(-1) * state
        + at::matmul(final_keys.transpose(-1, -2), writes);
  }
  return state;
}
}  // namespace

TORCH_LIBRARY(gradpert_kda_diagnostic, m) {
  m.def("final_state(Tensor k, Tensor v, Tensor log_decay, Tensor beta, "
        "Tensor? initial_state, int chunk_size) -> Tensor");
}

// Inner ATen calls retain their ordinary autograd and autocast dispatch.
// This is a composite implementation, not a hand-written backward formula.
TORCH_LIBRARY_IMPL(gradpert_kda_diagnostic, CompositeImplicitAutograd, m) {
  m.impl("final_state", final_state);
}
