import io
p = 'src/graphics/host_gpu/renderer/pipeline/pipelineCache.h'
s = io.open(p, encoding='utf-8').read().replace(
    'ProgramKey                             m_lookup_key;',
    'static thread_local ProgramKey         m_lookup_key;')
io.open(p, 'w', encoding='utf-8', newline='').write(s)
p = 'src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp'
s = io.open(p, encoding='utf-8').read()
s = s.replace('namespace Libs::Graphics {\n\nnamespace {',
              'namespace Libs::Graphics {\n\nthread_local PipelineCache::ProgramKey PipelineCache::m_lookup_key;\n\nnamespace {')
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('ok')
