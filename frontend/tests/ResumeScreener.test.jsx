import { describe, it, expect, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ResumeScreener from '../src/components/ResumeScreener.jsx';

const SCREEN = {
  category: 'Data Science',
  confidence: 0.82,
  topCategories: [
    { category: 'Data Science', probability: 0.82 },
    { category: 'Business Analyst', probability: 0.1 },
  ],
  skills: ['python', 'sql'],
};
const MATCH = { score: 0.62, cosine: 0.5, skillMatch: 0.8, matchedSkills: ['python'], missingSkills: ['statistics'] };

function setup(api) {
  const user = userEvent.setup();
  render(<ResumeScreener api={api} />);
  return user;
}

describe('ResumeScreener', () => {
  it('asks for a resume before calling the API', async () => {
    const api = { screenResume: vi.fn(), matchResume: vi.fn() };
    const user = setup(api);
    await user.click(screen.getByRole('button', { name: /analyse resume/i }));
    expect(screen.getByRole('alert')).toHaveTextContent('Paste a resume first.');
    expect(api.screenResume).not.toHaveBeenCalled();
  });

  it('shows the predicted category and skills', async () => {
    const api = { screenResume: vi.fn().mockResolvedValue(SCREEN), matchResume: vi.fn() };
    const user = setup(api);
    await user.type(screen.getByLabelText('Resume text'), 'Python and SQL analyst');
    await user.click(screen.getByRole('button', { name: /analyse resume/i }));

    const result = await screen.findByTestId('screen-result');
    expect(within(result).getByText('Data Science', { selector: 'strong' })).toBeInTheDocument();
    expect(within(result).getByText('Confidence 82%')).toBeInTheDocument();
    expect(within(result).getAllByRole('progressbar')).toHaveLength(2);
    expect(within(screen.getByRole('list', { name: 'Detected skills' })).getAllByRole('listitem')).toHaveLength(2);
    expect(api.screenResume).toHaveBeenCalledWith('Python and SQL analyst');
    expect(api.matchResume).not.toHaveBeenCalled();
    expect(screen.queryByTestId('match-result')).not.toBeInTheDocument();
  });

  it('also scores the job match when a job description is given', async () => {
    const api = { screenResume: vi.fn().mockResolvedValue(SCREEN), matchResume: vi.fn().mockResolvedValue(MATCH) };
    const user = setup(api);
    await user.type(screen.getByLabelText('Resume text'), 'resume');
    await user.type(screen.getByLabelText(/job description/i), 'job');
    await user.click(screen.getByRole('button', { name: /analyse resume/i }));

    const match = await screen.findByTestId('match-result');
    expect(match).toHaveTextContent('Job match: 62%');
    expect(match).toHaveTextContent('Strong match');
    expect(within(screen.getByRole('list', { name: 'Missing skills' })).getByText('statistics')).toBeInTheDocument();
    expect(api.matchResume).toHaveBeenCalledWith('resume', 'job');
  });

  it('shows API errors', async () => {
    const api = { screenResume: vi.fn().mockRejectedValue(new Error('Server down')), matchResume: vi.fn() };
    const user = setup(api);
    await user.type(screen.getByLabelText('Resume text'), 'resume');
    await user.click(screen.getByRole('button', { name: /analyse resume/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Server down');
    expect(screen.queryByTestId('screen-result')).not.toBeInTheDocument();
  });

  it('disables the button while loading', async () => {
    let resolve;
    const api = { screenResume: vi.fn(() => new Promise((r) => { resolve = r; })), matchResume: vi.fn() };
    const user = setup(api);
    await user.type(screen.getByLabelText('Resume text'), 'resume');
    await user.click(screen.getByRole('button', { name: /analyse resume/i }));
    expect(screen.getByRole('button', { name: /analysing/i })).toBeDisabled();
    resolve(SCREEN);
    expect(await screen.findByRole('button', { name: /analyse resume/i })).toBeEnabled();
  });
});
