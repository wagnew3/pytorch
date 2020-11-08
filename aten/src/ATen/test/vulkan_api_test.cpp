#ifdef USE_VULKAN_API

#include <gtest/gtest.h>
#include <ATen/ATen.h>

// TODO: These functions should move to a common place.

namespace {

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor>& inputs) {
  float maxValue = 0.0f;

  for (const auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }

  return diff.abs().max().item<float>() < (2e-6 * maxValue);
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.0f;
}

} // namespace

namespace {

TEST(VulkanAPITest, adaptive_avg_pool2d) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu = at::rand({5, 7, 47, 31}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::adaptive_avg_pool2d(in_cpu, {3, 3});
  const auto out_vulkan = at::adaptive_avg_pool2d(in_cpu.vulkan(), {3, 3});

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, add) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a_cpu, b_cpu, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_vulkan, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << c_cpu << std::endl;
    std::cout << "Got:\n" << c_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, add_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << a_cpu << std::endl;
    std::cout << "Got:\n" << a_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, add_scalar) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::add(a_cpu, b_scalar, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_scalar, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << c_cpu << std::endl;
    std::cout << "Got:\n" << c_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, add_scalar_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.add_(b_scalar, 2.1f);
  a_vulkan.add_(b_scalar, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << a_cpu << std::endl;
    std::cout << "Got:\n" << a_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, addmm) {
  if (!at::is_vulkan_available()) {
    return;
  }

  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  const auto bias_cpu = at::rand({179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  const auto bias_vulkan = bias_cpu.vulkan();
  const auto m1_vulkan = m1_cpu.vulkan();
  const auto m2_vulkan = m2_cpu.vulkan();
  const auto out_vulkan = at::addmm(bias_vulkan, m1_vulkan, m2_vulkan, beta, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, avg_pool2d) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu = at::rand({3, 19, 43, 79}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::avg_pool2d(in_cpu, {5, 3}, {1, 2}, {2, 0}, true);
  const auto out_vulkan = at::avg_pool2d(in_cpu.vulkan(), {5, 3}, {1, 2}, {2, 0}, true);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, clamp) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const float min_value = 0.2f;
  const float max_value = 0.8f;

  const auto out_cpu = at::clamp(in_cpu, min_value, max_value);
  const auto out_vulkan = at::clamp(in_vulkan, min_value, max_value);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, clamp_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  const float min_value = 0.2f;
  const float max_value = 0.8f;

  cpu.clamp_(min_value, max_value);
  vulkan.clamp_(min_value, max_value);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << cpu << std::endl;
    std::cout << "Got:\n" << vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, conv2d) {
  if (!at::is_vulkan_available()) {
    return;
  }

  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 1};
  constexpr std::array<int64_t, 2u> padding{0, 0};
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
        batches,
        channels,
        width,
        height,
      };
    }
  } input {1, 4, 3, 3};

  constexpr struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
        output_channels,
        input_channels,
        width,
        height,
      };
    }
  } weights {2, input.channels, 3, 3};

  const auto input_cpu = at::ones(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu = at::ones(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::zeros({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  const auto output_cpu = at::conv2d(
      input_cpu,
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  const auto output_vulkan = at::conv2d(
      input_cpu.vulkan(),
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  const bool check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << output_cpu << std::endl;
    std::cout << "Got:\n" << output_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, conv2d_depthwise) {
  if (!at::is_vulkan_available()) {
    return;
  }

  constexpr int64_t groups = 7;
  constexpr std::array<int64_t, 2u> stride{1, 3};
  constexpr std::array<int64_t, 2u> padding{2, 0};
  constexpr std::array<int64_t, 2u> dilation{1, 2};

  constexpr struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
        batches,
        channels,
        width,
        height,
      };
    }
  } input {1, groups, 137, 199};

  constexpr struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
        output_channels,
        input_channels,
        width,
        height,
      };
    }
  } weights {groups, 1, 17, 7};

  const auto input_cpu = at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu = at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::rand({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  const auto output_cpu = at::conv2d(
      input_cpu,
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  const auto output_vulkan = at::conv2d(
      input_cpu.vulkan(),
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  const bool check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << output_cpu << std::endl;
    std::cout << "Got:\n" << output_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, copy) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto cpu = at::rand({13, 17, 37, 19}, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  const auto check = exactlyEqual(cpu, vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << cpu << std::endl;
    std::cout << "Got:\n" << vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, empty) {
  if (!at::is_vulkan_available()) {
    return;
  }

  ASSERT_NO_THROW(at::empty({1, 17, 41, 53}, at::device(at::kVulkan).dtype(at::kFloat)));
}

TEST(VulkanAPITest, mean) {
  const auto in_cpu = at::rand({5, 3, 9, 9}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::mean(in_cpu, {-1, -2}, false);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::mean(in_vulkan, {-1, -2}, false);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, mean_keep_dim) {
  const auto in_cpu = at::rand({10, 3, 21, 21}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::mean(in_cpu, {-1, -2}, true);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::mean(in_vulkan, {-1, -2}, true);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, mm) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto m1_cpu = at::rand({241, 313}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({313, 193}, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = m1_cpu.mm(m2_cpu);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto m2_vulkan = m2_cpu.vulkan();
  const auto out_vulkan = m1_vulkan.mm(m2_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, mul_scalar) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({17, 213, 213, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::mul(a_cpu, b_scalar);
  const auto c_vulkan = at::mul(a_vulkan, b_scalar);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << c_cpu << std::endl;
    std::cout << "Got:\n" << c_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, mul_scalar_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.mul_(b_scalar);
  a_vulkan.mul_(b_scalar);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << a_cpu << std::endl;
    std::cout << "Got:\n" << a_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, reshape) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu = at::rand({47, 11, 83, 97}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const std::array<int64_t, 2> shape{47 * 83, 11 * 97};

  const auto out_cpu = at::reshape(in_cpu, shape);
  const auto out_vulkan = at::reshape(in_vulkan, shape);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
  std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, reshape_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto cpu = at::rand({59, 41, 19, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  const std::array<int64_t, 3> shape{59, 41 * 67, 19};

  cpu.reshape(shape);
  vulkan.reshape(shape);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << cpu << std::endl;
    std::cout << "Got:\n" << vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, upsample_nearest2d) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu = at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::upsample_nearest2d(in_cpu, {4, 6});

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::upsample_nearest2d(in_vulkan, {4, 6});

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

} // namespace

#endif /* USE_VULKAN_API */
