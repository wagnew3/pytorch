#include <torch/csrc/jit/cuda/cuda.h>

namespace torch {
namespace jit {

c10::intrusive_ptr<CUDAEvent> CUDAStream::recordEvent(c10::intrusive_ptr<CUDAEvent> event) {
  if (!event) {
    event = c10::make_intrusive<CUDAEvent>();
  }

  event->recordInternal(this);
  return event;
}

void CUDAStream::waitEvent(c10::intrusive_ptr<CUDAEvent> event) {
  event->event_->block(*stream_);
}

void CUDAStream::waitStream(c10::intrusive_ptr<CUDAStream> stream) {
  auto ev = c10::make_intrusive<CUDAEvent>();
  stream->recordEvent(ev);
  waitEvent(ev);
}

void CUDAEvent::record(c10::intrusive_ptr<CUDAStream> stream) {
  event_->record(*stream->stream_);
}

void CUDAEvent::recordInternal(CUDAStream *stream) {
  event_->record(*stream->stream_);
}

void CUDAEvent::wait(c10::intrusive_ptr<CUDAStream> stream) {
  event_->block(*stream->stream_);
}

/*
TORCH_LIBRARY(cuda, m) {
  auto stream_class = m.class_<torch::jit::CUDAStream>("Stream").def(torch::init<int64_t, int64_t>());
  auto event_class = m.class_<torch::jit::CUDAEvent>("Event").def(torch::init<bool, bool, bool>());

  stream_class.def("query", &CUDAStream::query)
    .def("record_event", &CUDAStream::recordEvent)
    .def("synchronize", &CUDAStream::synchronize)
    .def("wait_event", &CUDAStream::waitEvent)
    .def("wait_stream", &CUDAStream::waitStream);

  event_class.def("elapsed_time", &CUDAEvent::elapsedTime)
    .def("ipc_handle", &CUDAEvent::ipcHandle)
    .def("query", &CUDAEvent::query)
    .def("record", &CUDAEvent::record)
    .def("synchronize", &CUDAEvent::synchronize)
    .def("wait", &CUDAEvent::wait);
};*/
}
}
