import { describe, it, expect, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CandidateRanking from '../src/components/CandidateRanking.jsx';

const RESULTS = [
  { rank: 1, id: 'Asha', category: 'Data Science', score: 0.55, matchedSkills: ['python', 'sql'] },
  { rank: 2, id: 'Ravi', category: 'HR', score: 0.05, matchedSkills: [] },
];

describe('CandidateRanking', () => {
  it('validates input', async () => {
    const api = { rankCandidates: vi.fn() };
    const user = userEvent.setup();
    render(<CandidateRanking api={api} />);
    await user.click(screen.getByRole('button', { name: /rank candidates/i }));
    expect(screen.getByRole('alert')).toHaveTextContent(/job description and at least one resume/);
    expect(api.rankCandidates).not.toHaveBeenCalled();
  });

  it('sends parsed candidates and renders the ranking table', async () => {
    const api = { rankCandidates: vi.fn().mockResolvedValue({ results: RESULTS }) };
    const user = userEvent.setup();
    render(<CandidateRanking api={api} />);
    await user.type(screen.getByLabelText('Job description'), 'Data scientist');
    await user.type(screen.getByLabelText(/^resumes/i), 'Name: Asha\npython sql\n---\nName: Ravi\npayroll');
    await user.click(screen.getByRole('button', { name: /rank candidates/i }));

    const rows = await screen.findAllByRole('row');
    expect(rows).toHaveLength(3); // header + 2
    expect(within(rows[1]).getByText('Asha')).toBeInTheDocument();
    expect(within(rows[1]).getByText('Strong match')).toBeInTheDocument();
    expect(within(rows[2]).getByText('Weak match')).toBeInTheDocument();
    expect(api.rankCandidates).toHaveBeenCalledWith('Data scientist', [
      { id: 'Asha', resumeText: 'python sql' },
      { id: 'Ravi', resumeText: 'payroll' },
    ]);
  });

  it('shows API errors', async () => {
    const api = { rankCandidates: vi.fn().mockRejectedValue(new Error('Too many candidates')) };
    const user = userEvent.setup();
    render(<CandidateRanking api={api} />);
    await user.type(screen.getByLabelText('Job description'), 'job');
    await user.type(screen.getByLabelText(/^resumes/i), 'resume');
    await user.click(screen.getByRole('button', { name: /rank candidates/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Too many candidates');
  });
});
