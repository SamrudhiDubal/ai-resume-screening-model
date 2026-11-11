import { describe, it, expect, vi } from 'vitest';
import { screenResume, matchResume, rankCandidates, BASE_URL } from '../src/api.js';

const okFetch = (data) => vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(data) });

describe('api client', () => {
  it('posts the resume to /api/screen', async () => {
    const fetchMock = okFetch({ category: 'HR' });
    await expect(screenResume('text', fetchMock)).resolves.toEqual({ category: 'HR' });
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE_URL}/api/screen`);
    expect(opts.method).toBe('POST');
    expect(JSON.parse(opts.body)).toEqual({ resumeText: 'text' });
  });

  it('sends match and rank payloads', async () => {
    const fetchMock = okFetch({});
    await matchResume('r', 'j', fetchMock);
    await rankCandidates('j', [{ id: 1, resumeText: 'r' }], 3, fetchMock);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ resumeText: 'r', jobDescription: 'j' });
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ jobDescription: 'j', candidates: [{ id: 1, resumeText: 'r' }], topK: 3 });
  });

  it('surfaces the API error message', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 400, json: () => Promise.resolve({ error: 'bad input' }) });
    await expect(screenResume('x', fetchMock)).rejects.toThrow('bad input');
  });

  it('falls back to the status code when the body is not JSON', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 502, json: () => Promise.reject(new Error('no json')) });
    await expect(screenResume('x', fetchMock)).rejects.toThrow('Request failed (502)');
  });

  it('explains network failures', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
    await expect(screenResume('x', fetchMock)).rejects.toThrow(/Cannot reach the screening API/);
  });
});
